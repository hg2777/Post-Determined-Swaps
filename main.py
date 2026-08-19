# -*- coding: utf-8 -*-
"""
Compounding (post-determined) interest-rate swap pricing and GIRR sensitivity.

Prices the post-determined swap book from the two-tab population workbook and
reconciles MtM and GIRR deltas to RiskWatch.

@author: E42656
"""

import pandas as pd

from linearInterpolation import load_curve_set
from simpleSwapPortfolio import SimpleSwapPortfolio
from simpleSwapSensitivity import (simpleswap_girr_for_portfolio,
                                   simpleswap_girr_delta_with_riskwatch,
                                   build_simpleswap_specs)
from simpleSwapDiagnostics import write_simpleswap_diagnostics

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
XLSX = 'curves.xlsx'                 # workbook holding the curve tabs

# Tabs of curves.xlsx that are NOT zero curves and must be skipped by the
# curve loader. 'FX rates' holds currency pairs against spot rates, so its
# first column is text, not day offsets.
NON_CURVE_TABS = ['Input_Transactional_Data_CSV', 'Input_Mkt_Data_CSV',
                  'RiskWatch-Results', 'FX rates']

INTERPOLATION_METHOD = 'linear'

# --- swap book inputs --------------------------------------------------------
# Step 1 prices the post-determined swap book and reconciles it to RiskWatch;
# step 2 runs the GIRR sensitivity analysis. Both run off one
# SimpleSwapPortfolio and write to a single workbook ('Swap_MtM' and
# 'GIRR_Delta' tabs).
SWAP_INPUT_XLSX        = 'Post_Determined_Input.xlsx'  # two-tab population workbook
SWAP_FIXED_SHEET       = 'Fixed_legs'  # fixed-leg tab (transposed: deal per column)
SWAP_FLOAT_SHEET       = 'Float_Legs'  # float-leg tab
# Curve-name resolution. The post-determined discount and index names
# ('EUR-STR', 'EUR-STR-ERX') match the curve tabs and the RiskWatch Risk
# Factor IDs as-is, so no automatic renaming is applied. This dict is only for
# any additional explicit overrides (none needed here).
SWAP_CURVE_ALIASES     = {}
SWAP_VALUATION_DATE    = '2025-12-31'
SWAP_RW_MTM_CSV        = 'frtb_sa_report.csv'  # FRTB SA report for the RiskWatch comparison; None to skip
SWAP_RW_INSTRUMENT_COL = 'Instrument ID'
SWAP_RW_MTM_COL        = 'Mark To Market'
# The workbook does not carry pay/receive; deals without an explicit direction
# fall back to this ('pay' = pay fixed) and are listed once in the run log.
SWAP_DEFAULT_POSITION  = 'pay'
# Reset-rate override file: 2 cols (deal id, last reset rate); supersedes the
# Float tab's Last Reset Rate for that deal. The real start date is an
# input on the Float tab and is never overridden. None to skip.
SWAP_RESET_CSV         = None
# Historical-fixings workbook for the post-determined reset: one tab per
# fixings series, column 1 = fixing dates, column 2 = rates (decimals).
# The Float tab's 'Historical Fixings' cell names the series; the aliases
# dict maps that name to the workbook tab when they differ.
SWAP_HIST_FIXINGS_XLSX    = 'Historical Fixing Curves.xlsx'
SWAP_HIST_FIXINGS_ALIASES = None
SWAP_OUT               = 'simpleswap_results.xlsx'    # single output workbook
SWAP_DIAGNOSTICS_OUT   = 'simpleswap_diagnostics.xlsx'  # per-swap analytics + curve sensitivity

# --- GIRR sensitivity grid ---------------------------------------------------
# FRTB GIRR vertices, hardcoded as DAYS counted from the valuation date and
# paired positionally with their display labels: 0.25Y <-> 90, ..., 30Y <->
# 10957. Change the two tuples together (same length, ascending days -- the
# tent shock assumes a monotonic grid). The same grid drives the GIRR deltas,
# the RiskWatch reconciliation labels and the diagnostics 'Scenario Curves'.
GIRR_TENOR_LABELS = ('0.25Y', '0.5Y', '1Y', '2Y', '3Y', '5Y',
                     '10Y', '15Y', '20Y', '30Y')
GIRR_TENOR_DAYS   = (90, 181, 365, 730, 1096, 1826, 3652, 5479, 7305, 10957)
GIRR_SHOCK        = 0.0001   # 1bp bump-and-reprice shock

# Troubleshooting: restrict the swap run to these DealNums. None -> whole book.
SWAP_ONLY_IDS = None


# ----------------------------------------------------------------------------
# Reconciliation summaries
# ----------------------------------------------------------------------------
def _print_rw_reconciliation(mtm):
    """Short RiskWatch reconciliation summary for the priced swap book."""
    if 'MtM-RiskWatch' in mtm.columns:
        m = mtm[mtm['MtM-UAT'].notna() & mtm['MtM-RiskWatch'].notna()
                & (mtm['MtM-RiskWatch'] != 0)]
        rel = (m['MtM-UAT'] / m['MtM-RiskWatch'] - 1.0).abs() * 100.0
        print("  priced            : {0}".format(int(mtm['MtM-UAT'].notna().sum())))
        print("  compared to RW    : {0}".format(len(m)))
        if len(m):
            print("  median |UAT/RW-1| : {0:.2f}%".format(rel.median()))
            print("  within 5% / 10%   : {0:.1f}% / {1:.1f}%".format(
                (rel <= 5).mean() * 100.0, (rel <= 10).mean() * 100.0))
    else:
        n = int(mtm['MtM'].notna().sum()) if 'MtM' in mtm.columns else len(mtm)
        print("  priced : {0}   (no RiskWatch report supplied -> MtM only)".format(n))


def _print_girr_reconciliation(girr):
    """Aggregate GIRR vs RiskWatch over all (swap, tenor, curve) cells."""
    col = '(Delta-UAT/RW-1)%'
    if col not in girr.columns:
        print("  no GIRR rows matched RiskWatch")
        return
    vals = pd.to_numeric(girr[col], errors='coerce').dropna().abs()
    print("  GIRR cells compared to RW : {0}".format(len(vals)))
    if len(vals):
        print("  median |UAT/RW-1|         : {0:.4f}%".format(vals.median()))
        print("  within 5% / 10%           : {0:.1f}% / {1:.1f}%".format(
            (vals <= 5).mean() * 100.0, (vals <= 10).mean() * 100.0))
    # Material UAT deltas with no RiskWatch match flag a curve-name mismatch
    # (our forecast/discount curve name != the report's Risk Factor ID).
    if 'Delta-RiskWatch' in girr.columns:
        uat = pd.to_numeric(girr['Delta-UAT'], errors='coerce').abs()
        unmatched = girr['Delta-RiskWatch'].isna() & (uat > 1e-6)
        n_un = int(unmatched.sum())
        if n_un:
            curves = sorted(girr.loc[unmatched, 'Curve'].unique())
            print("  WARNING: {0} nonzero UAT cell(s) had no RW match on "
                  "curve(s) {1}".format(n_un, curves))


def price_simpleswaps(curves):
    """Swap book run, in two steps off one SimpleSwapPortfolio:

        Step 1 : price the post-determined swap book and compare MtM to RiskWatch
        Step 2 : GIRR delta sensitivities (split by Discount Curve / Curve Index)

    Both write to a single workbook ('Swap_MtM' and 'GIRR_Delta' tabs).
    """
    port = SimpleSwapPortfolio(curves, SWAP_INPUT_XLSX, SWAP_FIXED_SHEET,
                               SWAP_FLOAT_SHEET, SWAP_VALUATION_DATE,
                               rw_mtm_csv=SWAP_RW_MTM_CSV,
                               rw_instrument_col=SWAP_RW_INSTRUMENT_COL,
                               rw_mtm_col=SWAP_RW_MTM_COL,
                               only_ids=SWAP_ONLY_IDS,
                               reset_csv=SWAP_RESET_CSV,
                               default_position=SWAP_DEFAULT_POSITION,
                               curve_aliases=SWAP_CURVE_ALIASES,
                               hist_fixings_xlsx=SWAP_HIST_FIXINGS_XLSX,
                               hist_fixings_aliases=SWAP_HIST_FIXINGS_ALIASES)

    # -- Step 1 : pricing + RiskWatch comparison -----------------------------
    print("=" * 95)
    print("SIMPLESWAP STEP 1/2 : pricing + RiskWatch MtM comparison")
    print("  workbook  : {0}   tabs: {1} / {2}".format(
        SWAP_INPUT_XLSX, SWAP_FIXED_SHEET, SWAP_FLOAT_SHEET))
    print("  valuation : {0}   RiskWatch report: {1}".format(
        SWAP_VALUATION_DATE, SWAP_RW_MTM_CSV))
    print("-" * 95)
    mtm = port.summary()                         # prices the book (+ RW reconciliation)
    _print_rw_reconciliation(mtm)

    # -- Step 2 : GIRR sensitivities + RiskWatch comparison ------------------
    print("=" * 95)
    print("SIMPLESWAP STEP 2/2 : GIRR delta sensitivities vs RiskWatch")
    print("  GIRR shock: per physical curve (single 1bp tent, all roles)")
    print("-" * 95)
    girr_roles = simpleswap_girr_for_portfolio(port, GIRR_TENOR_DAYS,
                                               GIRR_TENOR_LABELS,
                                               shock=GIRR_SHOCK)
    # Collapse the per-role deltas onto their physical curve and attach the
    # RiskWatch comparison, matched on DealNum + tenor + curve.
    girr = simpleswap_girr_delta_with_riskwatch(girr_roles, SWAP_RW_MTM_CSV)
    print("  GIRR rows : {0}   curves: {1}   tenors: {2}".format(
        len(girr), sorted(girr['Curve'].unique()),
        list(pd.unique(girr['Tenor']))))
    if SWAP_RW_MTM_CSV:
        _print_girr_reconciliation(girr)

    with pd.ExcelWriter(SWAP_OUT) as xl:
        mtm.to_excel(xl, sheet_name='Swap_MtM', index=False, na_rep='N/A')
        girr.to_excel(xl, sheet_name='GIRR_Delta', index=False, na_rep='N/A')

    print("=" * 95)
    print("written to {0}  (tabs: Swap_MtM, GIRR_Delta)".format(SWAP_OUT))
    print("=" * 95)

    # Per-swap diagnostics workbook (both-leg analytics + altered-curve
    # sensitivity), to trace any MtM / GIRR differences against RiskWatch.
    write_simpleswap_diagnostics(SWAP_DIAGNOSTICS_OUT, port.curves,
                                 build_simpleswap_specs(port),
                                 port.valuation_date,
                                 GIRR_TENOR_DAYS, GIRR_TENOR_LABELS,
                                 method=INTERPOLATION_METHOD,
                                 shock=GIRR_SHOCK,
                                 to_eur=port._to_eur)

    return mtm, girr


def main():
    curves = load_curve_set(XLSX, exclude=NON_CURVE_TABS,
                            method=INTERPOLATION_METHOD)
    print("Curves loaded from: {0}".format(XLSX))
    print("Interpolation method: {0}\n".format(INTERPOLATION_METHOD))
    return price_simpleswaps(curves)


if __name__ == "__main__":
    main()