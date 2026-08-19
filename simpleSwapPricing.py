# -*- coding: utf-8 -*-
"""
Vanilla post-determined interest rate swaps.

@author: E42656
"""

import re

import numpy as np
import pandas as pd
from dateutil.relativedelta import MO, TH
from pandas.tseries.offsets import DateOffset
from pandas.tseries.holiday import (AbstractHolidayCalendar, Holiday,
                                    GoodFriday, EasterMonday, nearest_workday,
                                    next_monday, next_monday_or_tuesday)

import resetRate


pd.set_option('display.max_columns', None)

# 'annual'     -> DF = 1 / (1 + z) ** (days / 365)   (matches Example.xlsx)
# 'continuous' -> DF = exp(-z * days / 365)
COMPOUNDING = 'annual'


# ---------------------------------------------------------------------------
# Payment-date holiday calendars: US federal, UK bank and EU TARGET holidays.
# Payment dates roll Following over weekends and the UNION of the three --
# same convention as the pre-determined pricer.
# ---------------------------------------------------------------------------
class _USHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=nearest_workday),
        Holiday('MLK Day', month=1, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday('Presidents Day', month=2, day=1, offset=DateOffset(weekday=MO(3))),
        Holiday('Memorial Day', month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Juneteenth', month=6, day=19, start_date='2021-06-19',
                observance=nearest_workday),
        Holiday('Independence Day', month=7, day=4, observance=nearest_workday),
        Holiday('Labor Day', month=9, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Columbus Day', month=10, day=1, offset=DateOffset(weekday=MO(2))),
        Holiday('Veterans Day', month=11, day=11, observance=nearest_workday),
        Holiday('Thanksgiving', month=11, day=1, offset=DateOffset(weekday=TH(4))),
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
    ]


class _UKHolidays(AbstractHolidayCalendar):
    rules = [
        Holiday('New Year', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May Bank Holiday', month=5, day=1,
                offset=DateOffset(weekday=MO(1))),
        Holiday('Spring Bank Holiday', month=5, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer Bank Holiday', month=8, day=31,
                offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day', month=12, day=26,
                observance=next_monday_or_tuesday),
    ]


class _EUHolidays(AbstractHolidayCalendar):
    '''EU TARGET closing days (fixed calendar days, no weekend observance).'''
    rules = [
        Holiday('New Year', month=1, day=1),
        GoodFriday,
        EasterMonday,
        Holiday('Labour Day', month=5, day=1),
        Holiday('Christmas', month=12, day=25),
        Holiday('Goodwill Day', month=12, day=26),
    ]


_HOLIDAY_RANGE = ('1990-01-01', '2099-12-31')
_HOLIDAYS = None


def _holiday_set():
    '''Union of US, UK and EU holidays over the working range, built once.'''
    global _HOLIDAYS
    if _HOLIDAYS is None:
        days = set()
        for cal in (_USHolidays(), _UKHolidays(), _EUHolidays()):
            for d in cal.holidays(pd.Timestamp(_HOLIDAY_RANGE[0]),
                                  pd.Timestamp(_HOLIDAY_RANGE[1])):
                days.add(pd.Timestamp(d).normalize())
        _HOLIDAYS = frozenset(days)
    return _HOLIDAYS


# Business Day Rule calendars: one physical calendar per Cal code, selected by
# the rule's (CalEUR/CalUSD/CalGBP) suffix rather than the deal currency.
_CAL_CLASSES = {'EUR': _EUHolidays, 'USD': _USHolidays, 'GBP': _UKHolidays}
_HOLIDAYS_BY_CAL = {}


def _holiday_set_for(cal):
    '''Holiday set for a single Business Day Rule calendar (EUR/USD/GBP),
    built once and cached. EUR -> EU TARGET, USD -> US federal, GBP -> UK
    bank. An unknown code contributes weekends only (empty holiday set).'''
    key = str(cal).strip().upper()
    if key not in _HOLIDAYS_BY_CAL:
        cls = _CAL_CLASSES.get(key)
        if cls is None:
            _HOLIDAYS_BY_CAL[key] = frozenset()
        else:
            days = set()
            for d in cls().holidays(pd.Timestamp(_HOLIDAY_RANGE[0]),
                                    pd.Timestamp(_HOLIDAY_RANGE[1])):
                days.add(pd.Timestamp(d).normalize())
            _HOLIDAYS_BY_CAL[key] = frozenset(days)
    return _HOLIDAYS_BY_CAL[key]


_BDR_RE = re.compile(
    r'(regular|modified)\s+following\s+(\d+)\s*-?\s*day\s*'
    r'\(\s*cal\s*(eur|usd|gbp)\s*\)', re.IGNORECASE)


def parse_business_day_rule(text):
    '''Parse a Business Day Rule cell of the form

        "Regular/Modified Following x-day (CalEUR/CalUSD/CalGBP)"

    into {'modified': bool, 'offset': int, 'calendar': 'EUR'|'USD'|'GBP'}.
    A blank / unparseable cell returns None; the pricer then falls back to the
    legacy Following roll on the US/UK/EU union (unchanged behaviour).'''
    s = u'{0}'.format(text).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'):
        return None
    m = _BDR_RE.search(s)
    if not m:
        return None
    return {'modified': m.group(1).lower() == 'modified',
            'offset': int(m.group(2)),
            'calendar': m.group(3).upper()}


def _act_act_isda(start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    
    if end <= start: 
        return 0
    total = 0.0
    cursor = start
    while cursor < end:
        next_year = pd.Timestamp(year=cursor.year + 1, month=1, day=1)
        seg_end = min(end, next_year)
        days_in_year = 366 if cursor.is_leap_year else 365
        total += (seg_end - cursor).days / float(days_in_year)
        cursor = seg_end
    return total

def _thirty_360_us(start, end):
    '''US (NASD) 30/360 day-count, expressed in years.

        if D1 == 31            -> D1 = 30
        if D2 == 31 and D1 == 30 -> D2 = 30

    (D1 == 30 already covers the case where D1 was 31 and clamped above.)
    '''
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    d1, d2 = start.day, end.day
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 == 30:
        d2 = 30
    return (360 * (end.year - start.year)
            + 30 * (end.month - start.month)
            + (d2 - d1)) / 360.0
    
def swap_year_fraction(valuation_date, payment_date, basis):
    valuation_date = pd.Timestamp(valuation_date)
    payment_date = pd.Timestamp(payment_date)
    
    days = (payment_date - valuation_date).days
    b = str(basis).strip().lower()
    
    if b in ('actual/360', 'act/360', 'actual360'):
        return days / 360.0
    elif b in ('actual/365', 'act/365', 'actual365'):
        return days / 365.0
    elif b in ('30/360', '30u/360', 'us 30/360', 'thirty/360',
               'european 30/360'):
        return _thirty_360_us(valuation_date, payment_date)
    elif b in ('actual/actual', 'act/act', 'actualactual'):
        return _act_act_isda(valuation_date, payment_date)
    return days / 365.0


class SimpleSwap(object):

    def __init__(self, curves, params):
        self.curves = curves
        self.params = params

        self.valuation = pd.Timestamp(params['valuation_date'])
        self.notional = float(params['notional'])

        # Discount and forecast (forward-projection) curves are independent.
        # A single 'curve' key still works for both (back-compat).
        single = params.get('curve')
        self.discount_curve = params.get('discount_curve', single)
        self.forecast_curve = params.get('forecast_curve', single)
        # optional spread over the discount yield (added to the discount
        # zero rate for BOTH legs); 0.0 = no spread.
        self.spread_over_yield = float(params.get('spread_over_yield', 0.0))
        # float-leg margin (Spread): added on top of the projected forward
        # rate in _forward_rate. 0.0 leaves a vanilla (no-spread) leg
        # unchanged.
        self.float_spread = float(params.get('float_spread', 0.0))

        # full working frames (helper columns retained for internal use)
        self.fixed_full = self._value_leg('fixed')
        self.float_full = self._value_leg('float')

    # ----------------------------------------------------------- curve / DF
    def _adjust_payment_date(self, dt):
        '''Following roll: move forward past weekends and US/UK/EU holidays.'''
        d = pd.Timestamp(dt).normalize()
        holidays = _holiday_set()
        while d.weekday() >= 5 or d in holidays:
            d = d + pd.Timedelta(days=1)
        return d

    def _roll_business_day(self, dt, bdr):
        '''Business Day Rule roll for a PAYMENT date.

        x=0 gives the next business day (Following); 'Modified' rolls back to
        the previous business day when that next business day falls in a new
        month, while 'Regular' keeps the new-month date. The 'x-day' offset
        then steps x further business days forward from that adjusted date
        (x=0 -> no step; x is never negative). The calendar is the rule's own
        Cal (EUR/USD/GBP), NOT the deal currency. bdr None -> the legacy
        Following roll on the US/UK/EU union, unchanged.

        NOTE: with x=0 (all current deals) the offset never fires, so the
        Modified test acts on the next business day exactly as specified. For
        x>0 the offset is applied AFTER the Modified anchor is fixed.'''
        d = pd.Timestamp(dt).normalize()
        if bdr is None:
            return self._adjust_payment_date(d)

        holidays = _holiday_set_for(bdr['calendar'])

        def is_bus(x):
            return x.weekday() < 5 and x not in holidays

        def following(x):
            while not is_bus(x):
                x = x + pd.Timedelta(days=1)
            return x

        def preceding(x):
            while not is_bus(x):
                x = x - pd.Timedelta(days=1)
            return x

        base = following(d)
        if bdr['modified'] and (base.year, base.month) != (d.year, d.month):
            base = preceding(d)              # last business day within d's month

        res = base
        for _ in range(int(bdr['offset'])):  # 'x-day' forward offset
            res = following(res + pd.Timedelta(days=1))
        return res

    def _days(self, dt):
        return (pd.Timestamp(dt) - self.valuation).days

    def _discount_factor(self, dt, curve=None, spread=0.0):
        d = self._days(dt)
        if d <= 0:
            return 1.0
        if curve is None:
            curve = self.discount_curve
        z = float(self.curves.rate(curve, d)) + float(spread)
        if str(COMPOUNDING).strip().lower().startswith('cont'):
            return float(np.exp(-z * d / 365.0))
        return 1.0 / (1.0 + z) ** (d / 365.0)

    def _forward_rate(self, start, end, basis):
        '''Forward implied by the FORECAST curve over [start, end].'''
        tau = swap_year_fraction(start, end, basis)
        if tau <= 0:
            return 0.0
        df_s = self._discount_factor(start, self.forecast_curve)
        df_e = self._discount_factor(end, self.forecast_curve)
        # float-leg margin (Spread) added on top of the projected forward.
        return (df_s / df_e - 1.0) / tau + self.float_spread

    def _first_period_is_reset(self, first_period_start):
        '''
        True when the FIRST floating period carries a KNOWN fixing (the last
        reset rate) rather than a projected forward: the period has already
        started on/before the valuation date. Only the first live period can be
        a reset; every later period is always projected.
        '''
        return pd.Timestamp(first_period_start) <= self.valuation
    

    def _compounded_reset_rate(self, start, end, accrual):
        '''Post-determined first-period reset rate:

            reset = (ObservedAcc * ForwardAcc - 1) / accrual

        observed historical fixings compounded to the valuation date, then the
        index curve's forward growth to the period end, expressed as a simple
        rate over the period's own accrual fraction. Dividing by the leg's
        accrual makes rate * accrual reproduce the total interest factor
        exactly, matching the workbook's Reset Rate UAT.

        The observed leg reads params['hist_fixings_nodes']: (dates, rates)
        from the historical-fixings workbook (loaded by the portfolio via
        resetRate.load_fixing_curves), NOT a curves.xlsx zero-curve tab. The
        fixings are observed history and are never shocked.'''
        nodes = self.params.get('hist_fixings_nodes')
        if nodes is None:
            raise ValueError(
                'first floating period is in progress but no historical '
                'fixings were supplied (Historical Fixings = {0!r}); set '
                'SWAP_HIST_FIXINGS_XLSX in main.py'.format(
                    self.params.get('hist_fixings_curve')))
        fix_dates, fix_rates = nodes
        # Index rate at the accrual end, read THROUGH self.curves.rate() so
        # any GIRR tenor shock on the index curve is applied. It is handed to
        # last_reset_rate as a single index node -- its forward leg only
        # samples that one offset.
        off = (pd.Timestamp(end) - self.valuation).days
        z = float(self.curves.rate(self.forecast_curve, off))
        reset, _observed, _forward = resetRate.last_reset_rate(
            self.valuation, start, end,
            fix_dates, fix_rates, [off], [z], accrual=accrual)
        return reset

    # -------------------------------------------------------------- schedule
    def _schedule_boundaries(self, term_years, leg='fixed'):
        '''Period boundaries generated FORWARD from the real start date to
        maturity, with stub placement keyed on the day-of-month of the
        start vs the maturity:

          float leg, day(start) < day(maturity): BACK stub -- regular periods
              run off the start day-of-month and the FINAL cashflow ends on
              maturity.
          float leg, day(start) > day(maturity): FRONT stub -- the first
              cashflow ends on the first date carrying the maturity
              day-of-month; regular periods then land on maturity exactly.
          fixed leg, and the day(start) == day(maturity) case on either leg:
              the coupon dates carry the maturity day-of-month from the first
              cashflow onward (no stub when the days already match).

        The FRONT-stub / maturity-day-of-month anchoring is generated by
        stepping down from maturity; the BACK-stub case steps up from the start
        and appends maturity as the last boundary. Both are equivalent to
        forward generation from the real start when the day-of-month matches.'''
        real_start = pd.Timestamp(self.params['real_start_date'])
        maturity = pd.Timestamp(self.params['maturity_date'])
        months = int(round(float(term_years) * 12))

        back_stub = (leg == 'float' and real_start.day < maturity.day)

        if back_stub:
            # regular periods off the start day-of-month; final period -> maturity
            boundaries = [real_start]
            i = 1
            while True:
                nxt = real_start + DateOffset(months=months * i)
                if nxt >= maturity:
                    break
                boundaries.append(nxt)
                i += 1
            boundaries.append(maturity)      # last (possibly long) period
            return boundaries

        # anchor coupon dates on the maturity day-of-month; a front stub, when
        # present (day(start) > day(maturity)), falls first.
        boundaries = [maturity]
        i = 1
        while True:
            prev = maturity - DateOffset(months=months * i)
            if prev <= real_start:
                break
            boundaries.append(prev)
            i += 1
        boundaries.append(real_start)        # first (possibly short) period
        boundaries.reverse()
        return boundaries


    def _build_leg(self, leg):
        '''
        Schedule for one leg. The discount/payment date always rolls to the
        next business day (Following). The accrual endpoints are rolled on
        the FIXED leg (coupons accrue between adjusted dates) and left on
        the unadjusted schedule on the FLOAT leg (RiskWatch anchors the
        float accrual and forward projection on the unadjusted schedule).
        '''
        if leg == 'fixed':
            term = self.params['fixed_term_years']
            prev = self.params.get(
                'last_fixed_payment',
                self.params['real_start_date']
            )
            bdr = self.params.get('fixed_business_day_rule')
        else:
            term = self.params['float_term_years']
            prev = self.params.get(
                'last_float_payment',
                self.params['real_start_date']
            )
            bdr = self.params.get('float_business_day_rule')

        # Accrual endpoints roll on the FIXED leg (coupons accrue between
        # adjusted dates) and stay on the unadjusted schedule on the FLOAT
        # leg. The workbook carries no per-leg flag, so this convention is
        # fixed here.
        adjust = (leg == 'fixed')
    
        boundaries = self._schedule_boundaries(term, leg)
    
        rows = []
        for i in range(len(boundaries) - 1):
            rows.append({
                'period_start': boundaries[i],      # unadjusted schedule start
                'period_end': boundaries[i + 1],    # unadjusted schedule end
                'accrual_start': boundaries[i],     # unadjusted base (anchors accrual + forward)
                # discount/payment date rolls under this leg's Business Day
                # Rule (Modified/Regular Following, x-day offset, own Cal); a
                # blank rule falls back to the legacy union Following.
                'payment_date': self._roll_business_day(boundaries[i + 1], bdr),
            })
    
        df = pd.DataFrame(rows)
    
        # keep only future cashflows
        df = df[df['period_end'] > self.valuation].reset_index(drop=True)
    
        if df.empty:
            return df
    
        # first live period is anchored on this leg's own previous pay date, so
        # the first live accrual spans exactly one period.
        df.loc[df.index[0], 'accrual_start'] = pd.Timestamp(prev)

        basis = self.params['fixed_basis'] if leg == 'fixed' \
            else self.params['float_basis']

        # Accrual endpoints actually used. On the FIXED leg (adjust True) they
        # roll with this leg's Business Day Rule -- the same roll as
        # payment_date -- so a coupon accrues over exactly the interval it is
        # discounted on (accr_end_used == payment_date); a blank rule falls
        # back to the legacy union Following inside _roll_business_day. On the
        # FLOAT leg (adjust False) they stay on the unadjusted schedule, and
        # the float forward in _value_leg is projected over these SAME
        # endpoints so the projected rate and the accrual cover one interval.
        if adjust:
            _accr_roll = lambda d: self._roll_business_day(d, bdr)
        else:
            _accr_roll = lambda d: pd.Timestamp(d).normalize()
        df['accr_start_used'] = df['accrual_start'].apply(_accr_roll)
        # The first live period's accrual START is the effective /
        # previous-payment anchor (set on df.index[0] above); it is a
        # fixed schedule anchor, NOT a schedule date to be rolled, so it
        # stays on the UNADJUSTED schedule -- only payment dates roll.
        # Without this, a forward-starting deal whose effective date
        # falls on a weekend/holiday has its first coupon shortened by
        # the Following roll (e.g. P1022216: Sun 2032-11-21 -> Mon,
        # dropping the first fixed period from 365 to 364 days).
        df.loc[df.index[0], 'accr_start_used'] = pd.Timestamp(
            df.loc[df.index[0], 'accrual_start']).normalize()
        df['accr_end_used'] = df['period_end'].apply(_accr_roll)
        df['accrual'] = df.apply(
            lambda r: swap_year_fraction(
                r['accr_start_used'], r['accr_end_used'], basis),
            axis=1)
    
        return df

    # ----------------------------------------------------------------- value
    def _value_leg(self, leg):
        df = self._build_leg(leg)
        if df.empty:
            return df

        basis = self.params['fixed_basis'] if leg == 'fixed' \
            else self.params['float_basis']
            
        df['days'] = df['payment_date'].apply(self._days)
        

        if leg == 'fixed':
            # both legs discount on the discount curve (+ optional spread over yield)
            df['discount_factor'] = df['payment_date'].apply(
            lambda d: self._discount_factor(
                d, self.discount_curve, self.spread_over_yield))
            df['rate'] = float(self.params['fixed_rate'])
        else:
            df['discount_factor'] = df['payment_date'].apply(
                lambda d: self._discount_factor(d, self.discount_curve))
            rates = []
            for i, (_, r) in enumerate(df.iterrows()):
                if i == 0 and self._first_period_is_reset(r['period_start']):
                        # post-determined: observed fixings compounded to the
                        # valuation date, then index-curve forward growth to end
                    rates.append(self._compounded_reset_rate(
                            r['accr_start_used'], r['accr_end_used'],
                            r['accrual']))
                else:
                    # not yet fixed -> project the forward over the SAME
                    # interval as the accrual
                    rates.append(self._forward_rate(
                        r['accr_start_used'], r['accr_end_used'], basis))
            df['rate'] = rates

        df['cash_flow'] = self.notional * df['rate'] * df['accrual']
        df['pv'] = df['cash_flow'] * df['discount_factor']
        return df

    # ------------------------------------------------------------------- pvs
    def fixed_leg_pv(self):
        return float(self.fixed_full['pv'].sum()) if not self.fixed_full.empty else 0.0

    def float_leg_pv(self):
        return float(self.float_full['pv'].sum()) if not self.float_full.empty else 0.0

    def _is_payer(self):
        '''True if we PAY fixed (payer / "paying" swap).'''
        pos = str(self.params.get('position', 'pay')).strip().lower()
        return pos in ('pay', 'payer', 'pay_fixed', 'paying')

    def npv(self):
        '''
        Dirty NPV from the holder's perspective.
            payer    : receive float, pay fixed  -> float - fixed
            receiver : receive fixed, pay float  -> fixed - float
        '''
        if self._is_payer():
            return self.float_leg_pv() - self.fixed_leg_pv()
        return self.fixed_leg_pv() - self.float_leg_pv()