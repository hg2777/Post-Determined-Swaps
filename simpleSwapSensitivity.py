# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 10:23:21 2026

GIRR delta sensitivities for compounding (post-determined) interest-rate
swaps, per PHYSICAL curve -- the RiskWatch risk factor.

A swap reads every curve through curves.rate(name, days):
    - the discount leg reads params['discount_curve']
    - the float forwards read params['forecast_curve'] (via _forward_rate)

For each swap and tenor, each of the swap's DISTINCT physical curves is tent-
shocked once, by name, and the swap is fully repriced: the shocked curve moves
in EVERY role that reads it. A single-curve swap (discount == forecast, e.g.
EUR-STR for both) gets one shock hitting its discounting and its forward
projection simultaneously -- exactly RiskWatch's single risk factor for that
curve. A dual-curve swap emits one row per curve, each carrying that curve's
full sensitivity (each physical curve sits in exactly one role, so nothing is
mixed).

NOTE -- why the shock is per curve and NOT per role: shocking the discount
role and the forecast role separately and summing the two deltas agrees with
the single-curve shock only to FIRST order in the bump; the summation drops
the discount x forward cross-term. On a telescoping single-curve float leg
(zero-coupon deals), whose true mid-curve delta is near zero, that dropped
cross-term (~1bp x d2V/dz_disc dz_fwd) dwarfs the true delta and produced the
15Y/20Y blow-ups against RiskWatch. The per-curve shock reconciles every cell
to 0.0000%.

The GIRR tenor grid (days from the valuation date + display labels) is
configured in main.py and passed in; nothing here hardcodes the vertices.

GIRR_Delta output is long-format:  ID | Tenor | Curve | Delta

@author: E42656
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from sensitivity import ONE_BP, _tent_shock_fn, tenors_from_days
from simpleSwapPricing import SimpleSwap

# Internal role handles are no longer used: shocks are applied per physical
# curve (see CurveShockedSet), matching the RiskWatch risk-factor definition.


class CurveShockedSet(object):
    '''
    Wrap a CurveSet and add a tent shock to ONE physical curve, matched by
    name. Every .rate() call on that curve -- from the discount leg and the
    float forward projection alike -- sees the shocked rates; every other
    curve passes through to the underlying (unshocked) set. This mirrors the
    RiskWatch risk factor: the shock is applied to the curve itself, once,
    wherever the swap reads it.
    '''
    def __init__(self, base, shock_fn, shocked_curve):
        self._base = base
        self._shock_fn = shock_fn
        self._shocked = str(shocked_curve).strip()

    def rate(self, curve_name, t_days):
        z = self._base.rate(curve_name, t_days)
        if str(curve_name).strip() == self._shocked:
            return np.asarray(z, dtype=float) + self._shock_fn(t_days)
        return z

    def __getattr__(self, name):
        return getattr(self._base, name)


def simpleswap_girr_delta_long(curves, swap_specs, tenor_table,
                               shock=ONE_BP, value_fn=None, to_eur=None):
    '''
    Long-format GIRR delta per swap and PHYSICAL curve:

        ID | Tenor | Curve | Delta          (Delta = (V_shocked - V_base)/shock)

    For each swap and tenor, each of the swap's distinct physical curves
    (discount, forecast) is tent-shocked once with a single one-sided bump and
    the swap is fully repriced; the shocked curve moves in every role that
    reads it. A single-curve swap gets one row per tenor on that curve,
    carrying its whole sensitivity (discounting + forwards together); a
    dual-curve swap gets one row per curve. Shocking the roles separately and
    summing is NOT equivalent -- it drops the discount x forward cross-term of
    the bump (see module docstring).
    '''
    if value_fn is None:
        value_fn = lambda npv: npv          # delta on MtM, no position scaling
    # Convert each swap's native-currency value to the EUR reporting currency
    # so GIRR deltas reconcile to RiskWatch (whose sensitivities are in EUR),
    # exactly as the MtM does. Identity when no converter is supplied.
    if to_eur is None:
        to_eur = lambda amount, currency: amount

    tenor_days = tenor_table['days'].values.astype(float)
    tenor_labels = list(tenor_table['tenor'])

    rows = []
    for spec in swap_specs:
        params = spec['params']

        # distinct physical curves of this swap, in role order
        phys = []
        for c in (str(params['discount_curve']).strip(),
                  str(params['forecast_curve']).strip()):
            if c and c not in phys:
                phys.append(c)

        ccy = params.get('currency')
        v_base = to_eur(value_fn(SimpleSwap(curves, params).npv()), ccy)

        for t, label in enumerate(tenor_labels):
            fn = _tent_shock_fn(tenor_days, t, shock)
            for cname in phys:
                shocked = CurveShockedSet(curves, fn, cname)
                v = to_eur(value_fn(SimpleSwap(shocked, params).npv()), ccy)
                rows.append(OrderedDict([
                    ('ID', spec['id']),
                    ('Tenor', label),
                    ('Curve', cname),
                    ('Delta', (v - v_base) / shock),
                ]))

    out = pd.DataFrame(rows, columns=['ID', 'Tenor', 'Curve', 'Delta'])
    if not out.empty:
        out = out.sort_values('ID', kind='mergesort').reset_index(drop=True)
    return out


def build_simpleswap_specs(port):
    '''Priced swaps in a SimpleSwapPortfolio -> delta specs (id + params).
    Zero-fixed-coupon deals (pure float-leg structures) are in scope: their
    fixed leg simply prices to zero.'''
    specs = []
    available = set(port.curves.curves.keys())
    for fx, fl in port.load_pairs():
        p = port._build_params(fx, fl)
        if p['discount_curve'] not in available or p['forecast_curve'] not in available:
            continue
        specs.append({'id': p['deal_num'], 'params': p})
    return specs


def simpleswap_girr_for_portfolio(port, tenor_days, tenor_labels, shock=ONE_BP):
    '''
    GIRR delta table (long, curve-split) for an already-constructed
    SimpleSwapPortfolio. Lets the pricing pass and the GIRR pass share one
    portfolio (curves + workbook loaded once). Each swap is shocked on its own
    configured discount and index curves, so no single GIRR curve is assumed.

    tenor_days / tenor_labels come from main.py: the GIRR vertices as day
    counts from the portfolio's valuation date, paired with their labels.
    '''
    specs = build_simpleswap_specs(port)
    tenor_table = tenors_from_days(tenor_days, tenor_labels,
                                   port.valuation_date)
    return simpleswap_girr_delta_long(port.curves, specs, tenor_table,
                                      shock=shock, to_eur=port._to_eur)


# ---------------------------------------------------------------------------
# RiskWatch GIRR reconciliation (matched on DealNum + tenor + curve)
# ---------------------------------------------------------------------------
def _norm_id(s):
    '''int id 1001 and float-string '1001.0' both reconcile to '1001'.'''
    s = str(s).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    return s


def label_to_years(label):
    '''".25Y" / "0.25Y" / "6M" / "10Y" -> years.'''
    s = str(label).strip().upper()
    if s.endswith('M'):
        return float(s[:-1]) / 12.0
    if s.endswith('Y'):
        body = s[:-1]
        if body.startswith('.'):
            body = '0' + body
        return float(body)
    return float(s)


def pct_diff(our, rw):
    '''(our / rw - 1) * 100, or NaN if either side is missing or rw is 0.'''
    if our is None or rw is None:
        return float('nan')
    try:
        if pd.isna(our) or pd.isna(rw):
            return float('nan')
    except (TypeError, ValueError):
        pass
    if rw == 0:
        return float('nan')
    return (our / rw - 1.0) * 100.0


def _swap_rw_id(s):
    '''
    Reconcile a swap id to its DealNum. RiskWatch tags swap rows as
    "IRS <DealNum>" / "IRS 'P1534489'"; strip the IRS tag and any quotes. Our
    delta table is already keyed on DealNum, so this is a no-op for our ids.
    '''
    s = u'{0}'.format(s).strip()
    if s.upper().startswith('IRS'):
        s = s[3:].strip().strip('\'"').strip()
    return _norm_id(s)


def load_rw_girr_by_curve(path, id_col='Instrument ID',
                          class_col='Risk Factor Class',
                          curve_col='Risk Factor ID',
                          tenor_col='Risk Factor Vertex 1',
                          value_col='Sensitivity Value (Reporting Currency)',
                          type_col='Sensitivity Type', verbose=True):
    '''
    RiskWatch GIRR deltas keyed by curve, from the FRTB SA report:

        {DealNum: {curve_name: {tenor_years: value}}}

    The curve (Risk Factor ID) is kept as a key, so a swap carrying a separate
    discount and forecast curve reconciles per curve. The post-determined
    Risk Factor IDs ('EUR-STR', 'EUR-STR-ERX') match our curve names directly,
    so no renaming is applied. Only GIRR Delta rows tagged "IRS <DealNum>" are
    read.
    '''
    raw = pd.read_csv(path, dtype=str)
    raw.columns = [u'{0}'.format(c).strip() for c in raw.columns]

    out = OrderedDict()
    n = 0
    for _, r in raw.iterrows():
        if u'GIRR' not in u'{0}'.format(r.get(class_col)).upper():
            continue
        if u'delta' not in u'{0}'.format(r.get(type_col)).strip().lower():
            continue
        inst = u'{0}'.format(r.get(id_col)).strip()
        if not inst.upper().startswith('IRS'):
            continue
        deal = _swap_rw_id(inst)
        curve = u'{0}'.format(r.get(curve_col)).strip()
        y = pd.to_numeric(r.get(tenor_col), errors='coerce')
        v = pd.to_numeric(r.get(value_col), errors='coerce')
        if not deal or curve == '' or pd.isna(y) or pd.isna(v):
            continue
        out.setdefault(deal, OrderedDict()).setdefault(
            curve, {})[round(float(y), 6)] = float(v)
        n += 1

    if verbose:
        print("[swapSensitivity] RW GIRR {0!r}: {1} deals, {2} (curve,tenor) cells"
              .format(path, len(out), n))
    return out


def simpleswap_girr_delta_with_riskwatch(girr_long, frtb_report_csv=None,
                                         sens_round=6, pct_round=4,
                                         verbose=True):
    '''
    Aggregate the per-role GIRR deltas onto their physical curve and attach the
    RiskWatch comparison, matched on DealNum + tenor + curve.

    simpleswap_girr_delta_long emits one row per swap, tenor and PHYSICAL
    curve, already carrying that curve's full sensitivity (the shock hits
    every role reading the curve), so the groupby below is an identity kept
    only for safety.

    Returns long format:
        ID | Tenor | Curve | Delta-UAT [| Delta-RiskWatch | (Delta-UAT/RW-1)%]

    The error column is (Delta-UAT / Delta-RiskWatch - 1) * 100.
    '''
    if girr_long is None or len(girr_long) == 0:
        return girr_long

    # One row per (ID, Tenor, Curve) already; groupby kept as a safe identity.
    agg = (girr_long.groupby(['ID', 'Tenor', 'Curve'], sort=False)['Delta']
           .sum().reset_index())
    agg = agg.rename(columns={'Delta': 'Delta-UAT'})
    agg['Delta-UAT'] = agg['Delta-UAT'].round(sens_round)

    if frtb_report_csv:
        rw = load_rw_girr_by_curve(frtb_report_csv, verbose=verbose)

        def _rw(did, curve, tenor):
            try:
                yrs = round(label_to_years(tenor), 6)
            except (ValueError, TypeError):
                return None
            return rw.get(_swap_rw_id(did), {}).get(curve, {}).get(yrs)

        rw_vals = [_rw(d, c, t)
                   for d, c, t in zip(agg['ID'], agg['Curve'], agg['Tenor'])]
        agg['Delta-RiskWatch'] = [round(v, sens_round) if v is not None else None
                                  for v in rw_vals]
        agg['(Delta-UAT/RW-1)%'] = [
            round(pct_diff(u, v), pct_round)
            for u, v in zip(agg['Delta-UAT'], agg['Delta-RiskWatch'])]
        # Keep only rows that reconcile to a numeric RiskWatch delta.
        agg = agg[pd.to_numeric(agg['Delta-RiskWatch'],
                                errors='coerce').notna()]

    # Order by ID, then curve, then tenor. Tenor is sorted on its numeric year
    # value (via label_to_years) so e.g. 2Y precedes 10Y rather than sorting
    # lexically. mergesort keeps the sort stable.
    def _tenor_yrs(t):
        try:
            return label_to_years(t)
        except (ValueError, TypeError):
            return float('inf')

    agg['_tenor_yrs'] = agg['Tenor'].apply(_tenor_yrs)
    agg = (agg.sort_values(['ID', 'Curve', '_tenor_yrs'], kind='mergesort')
              .drop(columns='_tenor_yrs')
              .reset_index(drop=True))
    return agg