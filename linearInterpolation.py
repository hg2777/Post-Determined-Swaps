# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:20:11 2026

Curve construction and interpolation.

@author: E42656
"""


import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

class Interpolation(object):

    LINEAR_ALIASES = ('linear', 'lin', 'l')
    CUBIC_ALIASES = ('cubic', 'cubicspline', 'cubic_spline', 'spline', 'c')

    def __init__(self, tenors_days, rates, method='linear'):
        x = np.asarray(tenors_days, dtype=float)
        y = np.asarray(rates, dtype=float)
        if x.size == 0:
            raise ValueError(
                    'no numeric (tenor, rate) rows found; check the '
                    'tenor/rate column indices and the header setting.')

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # CubicSpline needs strictly increasing x; drop duplicate tenors.
        # Built on the sorted arrays -- a mask from the unsorted tenors
        # would drop the wrong rows on any tab not already in ascending
        # tenor order.
        keep = np.concatenate(([True], np.diff(x) > 0))
        self.x = x[keep]
        self.y = y[keep]

        self.method = str(method).strip().lower()

        if self.method in self.CUBIC_ALIASES:
            if len(self.x) < 2:
                raise ValueError(
                        'Cubic interpolation needs at least 2 distinct tenors.')
            from scipy.interpolate import CubicSpline
            self._spline = CubicSpline(self.x, self.y)
        elif self.method in self.LINEAR_ALIASES:
            self._spline = None
        else:
            raise ValueError(
                    "Unknown interpolation method '{0}'. "
                    "Use 'linear' or 'cubic'.".format(method))

    def __call__(self, t_days):
        if self._spline is None:
            return np.interp(t_days, self.x, self.y)

        t = np.clip(np.asarray(t_days, dtype=float), self.x[0], self.x[-1])
        result = self._spline(t)

        if np.isscalar(t_days):
            return float(result)
        return result

def curve_from_frame(df, tenor_col=0, rate_col=1, method='linear'):
    sub = df.iloc[:, [tenor_col, rate_col]].copy()
    sub.columns = ['tenor', 'rate']
    sub['tenor'] = pd.to_numeric(sub['tenor'], errors='coerce')
    sub['rate'] = pd.to_numeric(sub['rate'], errors='coerce')
    sub = sub.dropna()
    return Interpolation(sub['tenor'].values, sub['rate'].values, method=method)


class CurveSet(object):
    def __init__(self, curves):
        self.curves = curves

    def rate(self, curve_name, t_days):
        key = str(curve_name).strip()

        if key not in self.curves:
            raise KeyError('Curve {0} not in curves. Current curves: {1}'.format(
                    key, list(self.curves)))

        return self.curves[key](t_days)


def _filter_tabs(sheet_names, exclude):
    exclude_lower = set(str(e).strip().lower() for e in exclude)
    return [s for s in sheet_names if s.strip().lower() not in exclude_lower]

def load_curve_set(path, curve_tabs=None, exclude=(), tenor_col=0, rate_col=1,
                   header=None, method='linear'):

    xls = pd.ExcelFile(path)
    if curve_tabs is None:
        curve_tabs = _filter_tabs(xls.sheet_names, exclude)

    curves = {}
    for name in curve_tabs:
        tab = pd.read_excel(path, sheet_name=name, header=header)
        try:
            curves[name.strip()] = curve_from_frame(tab, tenor_col,
                   rate_col, method=method)
        except ValueError as exc:
            raise ValueError(u"curve tab '{0}' in {1}: {2}".format(
                    name, path, exc))
    return CurveSet(curves)