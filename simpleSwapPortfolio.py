# -*- coding: utf-8 -*-
"""
Multiple COMPOUNDING (post-determined, single-curve) IRS pricing from the
two-tab population workbook, reconciled against the RiskWatch FRTB SA report.

Reads one workbook (Post_Determined_Input1.xlsx) holding two TRANSPOSED tabs
-- field labels down column A, one deal per column -- matched on the deal
number embedded in the Name row:

  * a FIXED-leg tab ('Fixed_legs',  Name = "Fixed Leg of P1534489")
  * a FLOAT-leg tab ('Float_Legs',  Name = "Float Leg of P1534489")

Scope (current):
  * vanilla swaps only -- the Variable Notional field carries a single
    constant notional; multi-step (amortizing) schedules are skipped
  * zero-fixed-coupon deals ARE in scope (the fixed leg prices to zero, so
    the deal is +/- the float leg PV by position)

Pricing is delegated entirely to simpleSwapPricing.SimpleSwap; this module
reads the workbook, applies the scope filter, matches the legs, translates
each pair into the params dict SimpleSwap understands, and (optionally)
compares our MtM to the 'Mark To Market' column of the FRTB SA report
(frtb_sa_report.csv), where swaps appear in 'Instrument ID' as "IRS <DealNum>".

Targets Python 2.7 (no f-strings, explicit float division, object base).

@author: E42656
"""

import re
from collections import OrderedDict

import pandas as pd
from pandas.tseries.offsets import DateOffset

from simpleSwapPricing import SimpleSwap, parse_business_day_rule
import resetRate

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

INPUT_XLSX = 'Post_Determined_Input1.xlsx'
FIXED_SHEET = 'Fixed_legs'
FLOAT_SHEET = 'Float_Legs'
VALUATION_DATE = '2025-12-31'

# FX rates for the EUR reporting-currency conversion live on the curve
# workbook's 'FX rates' tab (pairs quoted EUR-base). The portfolio auto-loads
# them from here when main.py does not pass an fx_rates dict, so the EUR
# conversion works even with an un-wired main.py.
FX_XLSX = 'curves.xlsx'
FX_SHEET = 'FX rates'

# RiskWatch FRTB SA report (set to None to skip the MtM comparison).
RW_MTM_CSV = None
RW_INSTRUMENT_COL = 'Instrument ID'
RW_MTM_COL = 'Mark To Market'

# Fixed-leg Coupon Rate is quoted in PERCENT (e.g. 2.91 -> 0.0291).
RATE_IS_PERCENT = True

# Float-leg Last Reset Rate is already a DECIMAL (e.g. 0.02083); no /100.
RESET_IS_PERCENT = False

# Pay/receive direction is NOT carried in the population workbook. Per-deal
# overrides ({DealNum -> 'pay' / 'receive'}) are set in main.py; any deal with
# neither an explicit workbook field nor an override falls back to
# DEFAULT_POSITION ('pay' = pay fixed / payer) and is listed once in the log.
POSITIONS = {}
DEFAULT_POSITION = 'pay'

# Map a workbook curve name to the curve-tab / RiskWatch risk-factor name.
# The post-determined index names ('EUR-STR', 'EUR-STR-ERX') ARE the curve-tab
# names, so no automatic renaming is applied; unmapped names pass through
# unchanged. Set in main.py; the pricing engine is not touched.
CURVE_ALIASES = {}

# Optional reset-rate override file (set to None to skip). Two columns:
#   deal id | last reset rate
# When supplied, it supersedes the Float tab's Last Reset Rate for that deal.
# Rate units follow RESET_IS_PERCENT. The real start date is an input on
# the Float tab and is never overridden.
RESET_CSV = None

# Historical-fixings workbook: one tab per fixings series (column 1 dates,
# column 2 rates). Feeds the post-determined reset's observed accumulation.
# None -> deals whose first period is in progress cannot be priced.
HIST_FIXINGS_XLSX = None
# Float-tab series name -> workbook tab name (e.g. 'EUR-STR' -> 'EUR_STR_ON').
HIST_FIXINGS_ALIASES = {}

# Strings that mean "no value" (blank cells, Excel error tokens).
_NA_TOKENS = ('', 'nan', 'none', 'na', 'n/a', 'nat', '#value!', '#n/a',
              '#ref!', '#div/0!', '#name?', '#num!', '#null!')


def _u(v):
    '''Coerce a header or cell to unicode without tripping Python 2.7's
    implicit ASCII decode. utf-8 first, latin-1 as a never-fail fallback.'''
    if isinstance(v, bytes):
        try:
            return v.decode('utf-8')
        except UnicodeDecodeError:
            return v.decode('latin-1')
    return u'{0}'.format(v)


def _norm_header(col):
    return re.sub(r'[^a-z0-9]', '', _u(col).lower())


def _norm_id(v):
    s = u'{0}'.format(v).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s


def _resolve_col(df, wanted):
    '''Find a column by normalised name (tolerant of case / trailing spaces).'''
    target = _norm_header(wanted)
    for c in df.columns:
        if _norm_header(c) == target:
            return c
    return None


# Field parsers
def _clean_num(x):
    '''Parse a numeric cell; blanks / Excel error tokens -> None.'''
    s = u'{0}'.format(x).replace('%', '').replace(',', '.').strip()
    if s.lower() in _NA_TOKENS:
        return None
    return float(s)


def to_rate(x, rate_is_percent=RATE_IS_PERCENT):
    '''Percent (or decimal) string -> decimal rate. Missing -> 0.0.'''
    v = _clean_num(x)
    if v is None:
        return 0.0
    return v / 100.0 if rate_is_percent else v


def to_reset_rate(x, rate_is_percent=RESET_IS_PERCENT):
    v = _clean_num(x)
    if v is None:
        return float('nan')
    return v / 100.0 if rate_is_percent else v


def term_to_years(term):
    '''"6-Months"/"12 Months"/"6M" -> 0.5 / 1.0 / 0.5 (years), for SimpleSwap.'''
    digits = re.findall(r'\d+', u'{0}'.format(term))
    if not digits:
        raise ValueError('Cannot read a term from {0!r}'.format(term))
    return int(digits[0]) / 12.0


def parse_schedule(raw):
    '''
    "{2051/01/02 32000000|...}" -> [(Timestamp, notional), ...] sorted by date.
    A vanilla swap has exactly one entry: (maturity, constant notional).
    '''
    s = u'{0}'.format(raw).strip().strip('{}').strip()
    pairs = []
    for chunk in s.split('|'):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        date = pd.Timestamp(parts[0])              # YYYY/MM/DD is unambiguous
        notional = float(parts[1]) if len(parts) > 1 else 0.0
        pairs.append((date, notional))
    pairs.sort(key=lambda p: p[0])
    if not pairs:
        raise ValueError('Empty notional schedule: {0!r}'.format(raw))
    return pairs


# ---------------------------------------------------------------------------
# Two-tab population workbook reader (transposed: field per row, deal per col)
# ---------------------------------------------------------------------------
_SHEET_CANON = {
    'name':                 'name',
    'type':                 'leg_type',
    'realstartdate':        'real_start_date',
    'discountcurve':        'discount_curve',
    'currency':             'currency',
    'variablenotional':     'notional_field',
    'term':                 'term',
    'underlyingcurveindex': 'curve_index',
    'lastresetrate':        'last_reset_rate',
    'couponrate':           'coupon_rate',
    'daycountbasis':        'day_count_basis',
    'maturitydate':         'maturity_date',
    'historicalfixings':    'hist_fixings',
    'businessdayrule':      'business_day_rule',
    'spreadoveryield':      'spread_over_yield',
    'spread':               'float_spread',
    # pay/receive, if the workbook is ever extended to carry it
    'payreceiveindicator':  'pay_receive',
    'payreceive':           'pay_receive',
    'position':             'pay_receive',
    'direction':            'pay_receive',
}


def _deal_num_from_name(name):
    '''"Float Leg of P1534489" / "P1534489" -> "P1534489".'''
    s = _u(name).strip()
    idx = s.lower().rfind('leg of ')
    if idx != -1:
        s = s[idx + len('leg of '):].strip()
    return _norm_id(s)


def _read_swap_sheet(path, sheet):
    '''
    Read one transposed leg tab into a list of per-deal dict records keyed by
    canonical field names. Column A holds the field labels; each further column
    is one deal. The top index row and any column without a Name are ignored.
    '''
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    if raw.shape[1] < 2:
        return []

    keys = []
    for lab in raw.iloc[:, 0]:
        if lab is None or (isinstance(lab, float) and pd.isna(lab)):
            keys.append(None)
        else:
            keys.append(_SHEET_CANON.get(_norm_header(lab)))

    records = []
    for j in range(1, raw.shape[1]):
        rec = {}
        for i, key in enumerate(keys):
            if key is not None:
                rec[key] = raw.iat[i, j]
        name = rec.get('name')
        if name is None or _u(name).strip().lower() in _NA_TOKENS:
            continue                       # empty trailing column
        rec['deal_num'] = _deal_num_from_name(name)
        records.append(rec)
    return records


def notional_from_field(raw):
    '''
    'Variable Notional' -> (notional, n_entries). A plain number (commas as
    thousands separators) is a single entry; a brace/pipe schedule string
    (amortizing) reports its entry count so the vanilla filter can skip
    multi-step notionals.
    '''
    s = _u(raw).strip()
    try:
        return float(s.replace(',', '')), 1
    except (ValueError, TypeError):
        sched = parse_schedule(s)
        return sched[0][1], len(sched)


def load_rw_swap_mtm(path, instrument_col=RW_INSTRUMENT_COL, mtm_col=RW_MTM_COL):
    '''
    {DealNum -> Mark to Market} from the FRTB SA report.

    Swap rows carry Instrument ID like "IRS P1534489" / "IRS '100002'"; the
    leading IRS tag (and any quotes) are stripped to recover the DealNum.
    Non-IRS rows (bonds etc.) are ignored.
    '''
    df = pd.read_csv(path, dtype=str)
    inst_c = _resolve_col(df, instrument_col)
    mtm_c = _resolve_col(df, mtm_col)
    if inst_c is None or mtm_c is None:
        raise KeyError(
            'FRTB report missing columns {0!r}/{1!r}. Found: {2}'.format(
                instrument_col, mtm_col, list(df.columns)))

    out = {}
    for _, r in df.iterrows():
        inst = u'{0}'.format(r[inst_c]).strip()
        if not inst.upper().startswith('IRS'):
            continue
        deal = _norm_id(inst[3:].strip().strip('\'"').strip())
        mtm = _clean_num(r[mtm_c])
        if deal and mtm is not None:
            out[deal] = mtm
    return out


# ---------------------------------------------------------------------------
# FX rates (reporting currency = EUR): convert each deal's native-currency
# leg PVs / MtM to EUR for the RiskWatch comparison.
# ---------------------------------------------------------------------------
def load_fx_rates(path, sheet='FX rates', verbose=True):
    '''{PAIR -> rate} from the FX rates tab of the curve workbook, e.g.
    {'EURUSD': 1.1745, 'EURCHF': 0.9304}. The pairs are quoted EUR-base
    (units of the quote currency per 1 EUR), so a quote-currency amount
    converts to EUR by DIVISION.

    Layout-tolerant: for each row, the first cell whose letters form a
    6-letter currency pair (e.g. 'EURUSD', or 'EUR/USD' -> 'EURUSD') is the
    pair, and the first non-zero numeric cell to its right (else anywhere in
    the row) is its rate. This copes with a header row and with the pair /
    rate sitting in any two columns. Rows with no 6-letter pair (headers,
    blanks) are skipped.'''
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    out = {}
    for _, row in raw.iterrows():
        vals = list(row)
        pair_idx = None
        for k, cell in enumerate(vals):
            if len(re.sub(r'[^A-Z]', '', _u(cell).upper())) == 6:
                pair_idx = k
                break
        if pair_idx is None:
            continue
        pair = re.sub(r'[^A-Z]', '', _u(vals[pair_idx]).upper())
        rate = None
        order = list(range(pair_idx + 1, len(vals))) + list(range(0, pair_idx))
        for k in order:
            try:
                v = _clean_num(vals[k])
            except (ValueError, TypeError):
                v = None
            if v is not None and v != 0:
                rate = v
                break
        if rate is not None:
            out[pair] = float(rate)
    if verbose:
        if out:
            print('[simpleSwapPortfolio] FX rates from {0!r} tab {1!r}: {2}'
                  .format(path, sheet, out))
        else:
            print('[simpleSwapPortfolio] WARNING: no FX pairs read from {0!r} '
                  'tab {1!r}. Raw tab follows:'.format(path, sheet))
            print(raw.to_string())
    return out


# Reset-rate override file: {DealNum -> last_reset_rate}. The real start
# date is an input on the Float tab of the population workbook and is NOT
# overridable here.
_RESET_ID_ALIASES = ('dealnum', 'dealnumber', 'dealid', 'swapid', 'swapnum',
                     'id', 'instrumentid')
_RESET_RATE_ALIASES = ('lastresetrate', 'resetrate')


def _resolve_any(df, aliases):
    '''First column whose normalised header matches one of `aliases`.'''
    for c in df.columns:
        if _norm_header(c) in aliases:
            return c
    return None


def load_reset_rates(path, rate_is_percent=RESET_IS_PERCENT):
    '''
    Read the reset-rate override file into {DealNum -> last_reset_rate}.

    Two columns are expected (case / punctuation tolerant): a deal id and a
    last reset rate. Rate units follow rate_is_percent. Rows without a usable
    deal id or rate are skipped. The real start date is an input on the
    Float tab of the population workbook and cannot be overridden here.
    '''
    df = pd.read_csv(path, dtype=str)
    id_c = _resolve_any(df, _RESET_ID_ALIASES)
    rate_c = _resolve_any(df, _RESET_RATE_ALIASES)
    if id_c is None or rate_c is None:
        raise KeyError(
            'Reset file {0!r} needs a deal-id and a reset-rate column. '
            'Found: {1}'.format(path, [_u(c) for c in df.columns]))

    out = {}
    for _, r in df.iterrows():
        deal = _norm_id(r[id_c])
        if deal.lower() in _NA_TOKENS:
            continue
        v = to_reset_rate(r[rate_c], rate_is_percent)
        if not pd.isna(v):
            out[deal] = float(v)
    return out


# Portfolio: load -> filter to vanilla -> match on DealNum -> price
class SimpleSwapPortfolio(object):

    def __init__(self, curves, input_xlsx=INPUT_XLSX, fixed_sheet=FIXED_SHEET,
                 float_sheet=FLOAT_SHEET, valuation_date=VALUATION_DATE,
                 rate_is_percent=RATE_IS_PERCENT, reset_is_percent=RESET_IS_PERCENT,
                 rw_mtm_csv=RW_MTM_CSV, rw_instrument_col=RW_INSTRUMENT_COL,
                 rw_mtm_col=RW_MTM_COL, only_ids=None, reset_csv=RESET_CSV,
                 positions=POSITIONS, default_position=DEFAULT_POSITION,
                 curve_aliases=CURVE_ALIASES,
                 hist_fixings_xlsx=HIST_FIXINGS_XLSX,
                 hist_fixings_aliases=HIST_FIXINGS_ALIASES,
                 fx_rates=None, fx_xlsx=FX_XLSX, fx_sheet=FX_SHEET):
        self.curves = curves
        self.input_xlsx = input_xlsx
        self.fixed_sheet = fixed_sheet
        self.float_sheet = float_sheet
        self.valuation_date = valuation_date
        # fixed coupon is quoted in percent; the float last reset rate is a
        # decimal -- carried as two independent unit flags.
        self.rate_is_percent = rate_is_percent
        self.reset_is_percent = reset_is_percent
        # pay/receive is absent from the workbook; explicit per-deal overrides
        # (from main.py) win, then any workbook field, then the default --
        # defaulted deals are reported once in price().
        self.positions = {_norm_id(k): u'{0}'.format(v).strip().lower()
                          for k, v in (positions or {}).items()}
        self.default_position = default_position
        self._defaulted_position = []
        # workbook curve name -> curve-tab / RiskWatch name (explicit only)
        self.curve_aliases = dict(curve_aliases) if curve_aliases else {}

        # FX rates for the EUR reporting-currency conversion (pair -> rate,
        # EUR-base). An explicit fx_rates dict wins; otherwise auto-load from
        # the curve workbook's FX tab so the conversion works even when
        # main.py does not pass one. A missing file/tab leaves it empty: EUR
        # deals still price and any non-EUR deal fails loudly in _to_eur.
        if fx_rates:
            self.fx_rates = dict(fx_rates)
        elif fx_xlsx:
            try:
                self.fx_rates = load_fx_rates(fx_xlsx, fx_sheet)
            except Exception as exc:
                print('[simpleSwapPortfolio] WARNING: could not load FX rates '
                      'from {0!r} tab {1!r}: {2}'.format(fx_xlsx, fx_sheet, exc))
                self.fx_rates = {}
        else:
            self.fx_rates = {}

        # optional reset-rate override file, loaded once
        #   {deal_num -> last_reset_rate}
        self.reset_csv = reset_csv
        self.reset_lookup = (load_reset_rates(reset_csv, reset_is_percent)
                             if reset_csv else {})

        # historical-fixings workbook, loaded once: {tab: (dates, rates)}
        self.hist_fixings_aliases = dict(hist_fixings_aliases or {})
        self.hist_fixings = (resetRate.load_fixing_curves(hist_fixings_xlsx)
                             if hist_fixings_xlsx else {})

        self.rw_mtm_csv = rw_mtm_csv
        self.rw_instrument_col = rw_instrument_col
        self.rw_mtm_col = rw_mtm_col

        # optional troubleshooting filter: restrict the book to these DealNums
        self.only_ids = (set(_norm_id(i) for i in only_ids)
                         if only_ids else None)

        self.swaps = OrderedDict()        # deal_num -> simpleSwapPricing.SimpleSwap
        self.skipped = []                 # (deal_num, reason)
        self.rw_mtm = {}                  # deal_num -> RiskWatch MtM
        self.results = None               # DataFrame, set by price()

    # -- scope filter --------------------------------------------------------
    @staticmethod
    def _is_vanilla(row):
        '''Vanilla = a single (constant) notional. The population workbook has
        no notional-at-start/end flags, so only the schedule arity is checked.'''
        try:
            return notional_from_field(row.get('notional_field'))[1] == 1
        except Exception:
            return False

    def _alias_curve(self, name):
        '''Resolve a workbook curve name to a curve-tab name. An explicit
        curve_aliases entry wins; everything else -- the post-determined index
        and discount names already match the curve tabs -- passes through.'''
        key = _u(name).strip()
        return self.curve_aliases.get(key, key)

    REPORTING_CCY = 'EUR'

    def _to_eur(self, amount, currency):
        '''Convert a monetary amount from the deal's native currency to the
        EUR reporting currency using the FX rates tab. EUR (and NaN/None)
        pass through unchanged; a EUR-base pair (EUR<CCY>) converts by
        division, the inverse pairing (<CCY>EUR) by multiplication. An
        unmapped non-EUR currency raises so a missing rate fails loudly
        instead of silently reporting native units.'''
        if amount is None or pd.isna(amount):
            return amount
        ccy = _u(currency).strip().upper()
        if not ccy or ccy == self.REPORTING_CCY:
            return float(amount)
        direct = self.REPORTING_CCY + ccy
        inverse = ccy + self.REPORTING_CCY
        if direct in self.fx_rates:
            return float(amount) / self.fx_rates[direct]
        if inverse in self.fx_rates:
            return float(amount) * self.fx_rates[inverse]
        raise KeyError(
            'No FX rate to convert {0} to {1}; FX pairs loaded: {2}. '
            'Add the pair to the FX rates tab.'.format(
                ccy, self.REPORTING_CCY, sorted(self.fx_rates)))

    def _resolve_position(self, fx, deal_num):
        '''Pay/receive direction, in priority order: the per-deal override
        dict from main.py, then an explicit workbook field (if one is ever
        added), then self.default_position (recorded for one warning).'''
        ov = self.positions.get(deal_num, '')
        if ov.startswith('p'):
            return 'pay'
        if ov.startswith('r'):
            return 'receive'
        s = _u(fx.get('pay_receive')).strip().upper()
        if s.startswith('P'):
            return 'pay'
        if s.startswith('R'):
            return 'receive'
        self._defaulted_position.append(deal_num)
        return self.default_position

    # -- inputs --------------------------------------------------------------
    def load_pairs(self):
        '''[(fixed_rec, float_rec), ...] for vanilla deals matched on DealNum.

        Each DealNum is loaded ONCE: if the workbook carries duplicate leg
        columns for the same deal (e.g. a deal pasted twice), the first
        occurrence wins and the repeats are reported as skipped. Without this
        guard a duplicated deal produces two identical specs, and the GIRR
        aggregation (grouped on ID + Tenor + Curve) sums them -- doubling the
        reported delta (a flat +100% against RiskWatch).'''
        fixed = _read_swap_sheet(self.input_xlsx, self.fixed_sheet)
        floating = _read_swap_sheet(self.input_xlsx, self.float_sheet)

        float_by_num = {}
        for r in floating:
            # first occurrence wins; duplicates are ignored (fixed-side dup
            # detection below reports the deal once)
            float_by_num.setdefault(r['deal_num'], r)

        pairs = []
        self.skipped = []
        seen = set()
        for fx in fixed:
            num = fx['deal_num']
            if self.only_ids is not None and num not in self.only_ids:
                continue                          # outside the troubleshooting scope
            if num in seen:
                self.skipped.append((num, 'duplicate column in workbook (first occurrence used)'))
                continue
            seen.add(num)
            if not self._is_vanilla(fx):
                self.skipped.append((num, 'fixed leg not vanilla (multi-step notional)'))
                continue
            fl = float_by_num.get(num)
            if fl is None:
                self.skipped.append((num, 'no matching float leg'))
            elif not self._is_vanilla(fl):
                self.skipped.append((num, 'float leg not vanilla (multi-step notional)'))
            else:
                pairs.append((fx, fl))
        return pairs

    def _build_params(self, fx, fl):
        '''Translate a matched fixed/float pair into a SimpleSwap dict.'''
        notional = notional_from_field(fx.get('notional_field'))[0]
        maturity = pd.to_datetime(fx['maturity_date'])
        real_start = pd.to_datetime(fx['real_start_date'])
        valuation = pd.Timestamp(self.valuation_date)

        # pay/receive is not in the population workbook (see _resolve_position)
        position = self._resolve_position(fx, fx['deal_num'])

        deal_num = fx['deal_num']
        deal_id = deal_num

        fixed_term_years = term_to_years(fx['term'])
        float_term_years = term_to_years(fl['term'])

        # spread over the discount yield (added to the discount zero rate for
        # both legs); from a 'Spread Over Yield' row when present, else 0.
        spread_over_yield = to_rate(
            fx.get('spread_over_yield', fl.get('spread_over_yield')),
            self.rate_is_percent)

        # Previous reset for a leg: the last schedule boundary on or before the
        # valuation date, generated with the SAME real start + k*term step
        # SimpleSwap uses. This equals the first live period's period_start, so
        # the first live accrual spans exactly one period instead of running
        # from the (possibly long-past) real start date. For a forward-starting
        # deal (real start > valuation) it is simply the real start date.
        def _prev_reset(term_years):
            months = int(round(float(term_years) * 12))
            prev = real_start
            i = 1
            while True:
                nxt = real_start + DateOffset(months=months * i)
                if nxt > valuation:
                    break
                prev = nxt
                i += 1
            return prev

        # Floating first-period reset rate (decimal) from the Float tab; the
        # optional override file supersedes it. SimpleSwap uses it -- via the
        # post-determined compounding -- only when the valuation date is past
        # the real start date; otherwise the period is projected forward.
        last_reset_rate = to_reset_rate(fl.get('last_reset_rate'),
                                        self.reset_is_percent)
        if deal_num in self.reset_lookup:
            last_reset_rate = self.reset_lookup[deal_num]

        return {
            # identification / reporting
            'id':                deal_id,
            'deal_num':          deal_num,
            'instrument_type':   _u(fx.get('leg_type', 'Compounding IRS')).strip()
                                 or 'Compounding IRS',
            'currency':          u'{0}'.format(fx.get('currency', '')).strip(),

            # pricing inputs consumed by simpleSwapPricing.SimpleSwap
            'valuation_date':    self.valuation_date,
            'notional':          notional,
            'position':          position,
            'real_start_date':   real_start,
            'maturity_date':     maturity,
            'fixed_rate':        to_rate(fx.get('coupon_rate'), self.rate_is_percent),
            'fixed_term_years':  fixed_term_years,
            'fixed_basis':       _u(fx.get('day_count_basis', '')).strip(),
            'float_term_years':  float_term_years,
            'float_basis':       _u(fl.get('day_count_basis', '')).strip(),
            # float-leg margin (Spread) added on top of the projected forward
            # rate in _forward_rate; percent units like the fixed rate.
            'float_spread':      to_rate(fl.get('float_spread'),
                                         self.rate_is_percent),
            # Business Day Rule per leg (Modified/Regular Following, x-day
            # offset, own Cal calendar). None when the cell is blank/absent ->
            # pricer falls back to the legacy union Following roll.
            'fixed_business_day_rule': parse_business_day_rule(
                fx.get('business_day_rule')),
            'float_business_day_rule': parse_business_day_rule(
                fl.get('business_day_rule')),
            'float_index':       u'{0}'.format(fl.get('curve_index', '')).strip(),
            'last_reset_rate':   last_reset_rate,
            'discount_curve':    self._alias_curve(fx.get('discount_curve', '')),
            'forecast_curve':    self._alias_curve(fl.get('curve_index', '')),
            # curve tab naming the historical fixings for the post-determined
            # observed accumulation; None when the workbook has no such row
            # (only needed once a deal's first period is in progress).
            'hist_fixings_curve': (_u(fl.get('hist_fixings', '')).strip()
                                   or None),
            # (dates, rates) from the fixings workbook for that series;
            # None when the deal names no series (forward-starting) or no
            # workbook was supplied -- the pricer then errors only if the
            # reset branch actually fires.
            'hist_fixings_nodes': resetRate.resolve_fixing_curve(
                self.hist_fixings, fl.get('hist_fixings', ''),
                self.hist_fixings_aliases) if self.hist_fixings else None,
            'spread_over_yield': spread_over_yield,
            'last_fixed_payment': _prev_reset(fixed_term_years),
            'last_float_payment': _prev_reset(float_term_years),
        }

    # -- pricing -------------------------------------------------------------
    def price(self):
        '''Price every compounding swap via SimpleSwap; return the summary.'''
        pairs = self.load_pairs()
        self._defaulted_position = []

        self.rw_mtm = {}
        if self.rw_mtm_csv:
            self.rw_mtm = load_rw_swap_mtm(self.rw_mtm_csv,
                                           self.rw_instrument_col,
                                           self.rw_mtm_col)
        compare = bool(self.rw_mtm_csv)

        self.swaps = OrderedDict()
        rows = []
        available = set(self.curves.curves.keys())
        for fx, fl in pairs:
            params = self._build_params(fx, fl)

            # skip cleanly when the deal cannot be priced here. Zero-coupon
            # deals ARE in scope: the fixed leg simply prices to zero and the
            # deal's MtM is +/- the float leg PV by position.
            missing = [c for c in (params['discount_curve'], params['forecast_curve'])
                       if c not in available]
            if missing:
                self.skipped.append((params['deal_num'],
                                     'curve(s) not loaded: {0}'.format(missing)))
                continue

            try:
                swap = SimpleSwap(self.curves, params)
                self.swaps[params['deal_num']] = swap
                fixed_pv, float_pv, mtm, err = (swap.fixed_leg_pv(),
                                                swap.float_leg_pv(),
                                                swap.npv(), '')
            except Exception as e:
                fixed_pv = float_pv = mtm = float('nan')
                err = str(e)

            # Report in the EUR reporting currency: convert this deal's
            # native-currency leg PVs and MtM to EUR before the RiskWatch
            # comparison (RiskWatch MtM is already EUR). EUR deals are
            # unchanged; Notional stays in the deal's own currency.
            ccy = params['currency']
            fixed_pv = self._to_eur(fixed_pv, ccy)
            float_pv = self._to_eur(float_pv, ccy)
            mtm = self._to_eur(mtm, ccy)

            row = OrderedDict([
                ('DealNum',      params['deal_num']),
                ('ID',           params['id']),
                ('Swap type',    params['instrument_type']),
                ('Currency',     params['currency']),
                ('Position',     'PAY fixed' if params['position'] == 'pay'
                                 else 'RECEIVE fixed'),
                ('Notional',     params['notional']),
                ('Fixed Leg PV', round(fixed_pv, 2)),
                ('Float Leg PV', round(float_pv, 2)),
            ])

            if compare:
                rw = self.rw_mtm.get(params['deal_num'], float('nan'))
                if pd.notna(mtm) and pd.notna(rw) and rw != 0:
                    pct = (mtm / rw - 1.0) * 100.0
                else:
                    pct = float('nan')
                row['MtM-UAT'] = round(mtm, 2)
                row['MtM-RiskWatch'] = round(rw, 2) if pd.notna(rw) else float('nan')
                row['(MtM-UAT/RW-1)%'] = round(pct, 4) if pd.notna(pct) else float('nan')
            else:
                row['MtM'] = round(mtm, 2)

            row['Error'] = err
            rows.append(row)

        out = pd.DataFrame(rows)
        if 'Error' in out.columns and (out['Error'] == '').all():
            out = out.drop(columns=['Error'])
        self.results = out

        if self._defaulted_position:
            print('NOTE: pay/receive not in the workbook; defaulted {0} deal(s) '
                  'to {1!r}: {2}'.format(len(self._defaulted_position),
                                         self.default_position,
                                         self._defaulted_position))
        if self.skipped:
            from collections import Counter
            reasons = Counter(r for _, r in self.skipped)
            print('Skipped {0} deal(s):'.format(len(self.skipped)))
            for reason, n in reasons.most_common():
                print('  {0:>4d}  {1}'.format(n, reason))

        return out

    def summary(self):
        '''The per-swap results DataFrame (prices on first call if needed).'''
        if self.results is None:
            self.price()
        return self.results