# -*- coding: utf-8 -*-
"""
Per-swap diagnostics workbook for the compounding (single-curve) IRS pricer.

Writes a second workbook (alongside swap_results.xlsx) whose two tabs let an
MtM / GIRR difference against RiskWatch be traced swap by swap and curve by
curve:

    'MtM Analytics'   : for each swap, the priced FIXED leg and FLOAT leg tables
                    (schedule dates, accruals, rates, discount factors, PVs)
                    with a header line, plus a small summary table carrying the
                    fixed/float leg PVs, MtM, notional and position.
    'Scenario Curves' : every curve that was altered to produce the GIRR deltas
                    (each swap's discount and forecast curve, de-duplicated),
                    one block per curve: the unchanged curve ('BaseRate'),
                    a parallel 1bp column and one column per GIRR tenor showing
                    the curve after that tenor's tent shock.

Targets Python 2.7 (no f-strings, .format(), object base classes).

@author: E42656
"""

import pandas as pd

from simpleSwapPricing import SimpleSwap
from sensitivity import (ONE_BP, build_sensitivity_table, get_curve_nodes,
                         tenors_from_days)

# Per-leg columns shown in the analytics tables, in order. Filtered to those
# the pricer actually produced, so it is safe if a column is absent.
LEG_COLS = ['period_start', 'period_end', 'accrual_start', 'payment_date',
            'accrual', 'rate', 'discount_factor', 'cash_flow', 'pv']

DATE_FMT = 'YYYY-MM-DD'


def _dates_only(frame):
    '''Return a copy with any datetime column reduced to a plain date.'''
    out = frame.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.date
    return out

def _put(xl, sheet, row, frame, header=True):
    '''Write a frame at `row` (date-only) and return the next free row index.'''
    _dates_only(frame).to_excel(xl, sheet_name=sheet, startrow=row,
                                index=False, header=header)
    return row + len(frame) + (1 if header else 0)

def swap_leg_tables(curves, params, to_eur=None):
    '''Price one swap and return (fixed-leg df, float-leg df, priced Swap).
    The monetary columns (cash_flow, pv) are converted to the EUR reporting
    currency via `to_eur` so the diagnostics tie out to the results
    workbook; to_eur=None leaves them in the deal's native currency.'''
    if to_eur is None:
        to_eur = lambda amount, currency: amount
    ccy = params.get('currency')
    swap = SimpleSwap(curves, params)

    def _leg(df):
        if df is None or len(df) == 0:
            return pd.DataFrame()
        cols = [c for c in LEG_COLS if c in df.columns]
        out = df[cols].copy()
        for c in ('cash_flow', 'pv'):
            if c in out.columns:
                out[c] = out[c].apply(lambda v: to_eur(v, ccy))
        return out

    return _leg(swap.fixed_full), _leg(swap.float_full), swap


def altered_curves(swap_specs):
    '''
    De-duplicated list of every curve that gets shocked across the book, in
    first-seen order: each swap's discount curve then its forecast curve.
    '''
    names = []
    for spec in swap_specs:
        params = spec['params']
        for role in ('discount_curve', 'forecast_curve'):
            nm = u'{0}'.format(params.get(role, '')).strip()
            if nm and nm not in names:
                names.append(nm)
    return names


def write_simpleswap_diagnostics(path, curves, swap_specs, valuation_date,
                           tenor_days, tenor_labels,
                           method='linear', shock=ONE_BP, to_eur=None,
                           verbose=True):
    '''
    Build the two-tab diagnostics workbook for the priced swap book.

    swap_specs : list of {'id', 'params'} (build_simpleswap_specs output).
    tenor_days / tenor_labels come from main.py, so the Scenario Curves tab
    shocks exactly the same grid as the GIRR sensitivity run.
    '''
    if not swap_specs:
        if verbose:
            print('[swapDiagnostics] no swaps to write; skipped {0}'.format(path))
        return path

    tenor_table = tenors_from_days(tenor_days, tenor_labels, valuation_date)

    # Report every monetary figure in the EUR reporting currency, exactly
    # as the results workbook does, so the two tie out for non-EUR deals;
    # identity when no converter is supplied.
    if to_eur is None:
        to_eur = lambda amount, currency: amount

    with pd.ExcelWriter(path, engine='openpyxl',
                        date_format=DATE_FMT, datetime_format=DATE_FMT) as xl:
        # --- Analytics tab : both legs + summary, per swap ------------------
        r = 0
        for spec in swap_specs:
            params = spec['params']
            sid = u'{0}'.format(spec.get('id', params.get('id', '')))
            ccy = params.get('currency')
            fixed_t, float_t, swap = swap_leg_tables(curves, params, to_eur)

            header = ('Swap {0}  ({1} {2})   position={3}  discount={4}  '
                      'forecast={5}'.format(
                          sid, params.get('currency', ''),
                          params.get('instrument_type', ''),
                          params.get('position', ''),
                          params.get('discount_curve', ''),
                          params.get('forecast_curve', '')))
            r = _put(xl, 'MtM Analytics', r,
                     pd.DataFrame([[header]]), header=False)

            r = _put(xl, 'MtM Analytics', r,
                     pd.DataFrame([['FIXED LEG']]), header=False)
            if len(fixed_t):
                r = _put(xl, 'MtM Analytics', r, fixed_t)
            r += 1

            r = _put(xl, 'MtM Analytics', r,
                     pd.DataFrame([['FLOAT LEG']]), header=False)
            if len(float_t):
                r = _put(xl, 'MtM Analytics', r, float_t)
            r += 1

            summary = pd.DataFrame(
                [[round(to_eur(swap.fixed_leg_pv(), ccy), 2),
                  round(to_eur(swap.float_leg_pv(), ccy), 2),
                  round(to_eur(swap.npv(), ccy), 2),
                  round(float(params['notional']), 2),
                  params.get('position', '')]],
                columns=['Fixed Leg PV', 'Float Leg PV', 'MtM',
                         'Notional', 'Position'])
            r = _put(xl, 'MtM Analytics', r, summary)
            r += 2

        # --- Sensitivity tab : one block per altered curve ------------------
        r = 0
        for name in altered_curves(swap_specs):
            nd, nr = get_curve_nodes(curves, name)
            table = build_sensitivity_table(valuation_date, nd, nr, tenor_table,
                                            method=method, shock=shock)
            r = _put(xl, 'Scenario Curves', r,
                     pd.DataFrame([['Curve: {0}'.format(name)]]), header=False)
            r = _put(xl, 'Scenario Curves', r, table)
            r += 2

    if verbose:
        print('[swapDiagnostics] wrote {0}  (tabs: Analytics, Sensitivity) '
              'for {1} swaps, {2} curves'.format(
                  path, len(swap_specs), len(altered_curves(swap_specs))))
    return path