import datetime as dt
# Defies a YieldCurve object with methods PV and FV

class YieldCurve(object):
    def __init__(self, rates, verbose=False):
        self.__rates__ = rates
        self.__verbose__ = verbose

    def get_rate(self, as_of, days):
        df = self.__rates__
        idx = df.index[df.index.get_loc(as_of, method='pad')]
        if days <= 15:
            r = df.loc[idx, 'FEDL01 Index']
        elif days <= 365:
            r = df.loc[idx, 'US0003M Index']
        elif days <= 365 * 2.5:
            r = df.loc[idx, 'GT2 Govt']
        elif days <= 365 * 4:
            r = df.loc[idx, 'GT3 Govt']
        else:
            r = df.loc[idx, 'GT5 Govt']
        return(r/100)

    def PV(self, dates, cfs):
        d = min(dates)
        days = (dates - d).days
        rates = days.map(lambda x: self.get_rate(d, x))
        if self.__verbose__:
            print('Days')
            print(days)
            print('Rates')
            print(rates)
        return((cfs / (1 + rates/2)**(2*days/365)).sum())

    def FV(self, dates, cfs):
        d = max(dates)
        days = (d - dates).days
        rates = days.map(lambda x: self.get_rate(d - dt.timedelta(days=x), x))
        if self.__verbose__:
            print('Days')
            print(days)
            print('Rates')
            print(rates)
        return((cfs * (1 + rates/2)**(2*days/365)).sum())