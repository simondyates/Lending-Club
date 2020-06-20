# Define a class to track a portfolio of LC loans; construct cash-flows for them; and calculate metrics
from collections import defaultdict
import numpy_financial as npf
import datetime as dt
import pandas as pd

class Portfolio:
    def __init__(self, universe, yc, rec_lag=0, lc_fee=0.01):
        self.__univ__ = universe # df of all the loan data we have
        self.__yc__ = yc # YieldCurve object
        self.rec_lag = rec_lag
        self.lc_fee = lc_fee
        self.ids = []
        self.amnts = []
        self.n_loans = defaultdict(int)
        self.payments = defaultdict(int)
        self.receipts = defaultdict(int)
        self.PVs = defaultdict(int)

    def __gen_cfs__(self, id, amnt):
        start = dt.datetime.strptime(self.__univ__.loc[id, 'issue_d'].astype('str'), '%Y%m%d')
        stop = dt.datetime.strptime(self.__univ__.loc[id, 'last_pymnt_d'].astype('str'), '%Y%m%d')
        self.n_loans[start] += 1
        self.payments[start] -= amnt
        ratio = amnt / self.__univ__.loc[id, 'funded_amnt']
        self.PVs[start] += ratio * self.__univ__.loc[id, 'PV']
        months = int((stop - start).days * 12 / 365)
        # adjust payments for LC's 1% fee
        pmts = ratio * (self.__univ__.loc[id, 'total_pymnt'] - self.__univ__.loc[id, 'recoveries']) * (1 - self.lc_fee)
        if months == 0:
            self.receipts[start] += pmts
        else:
            d = start
            for i in range(1, months):
                d = d + dt.timedelta(days=31)
                d = dt.datetime(d.year, d.month, 1)
                self.receipts[d] += pmts / months
        recs = ratio * self.__univ__.loc[id, 'recoveries'] # No LC fee on recoveries
        if recs > 0:
            d = stop + dt.timedelta(days = 31*self.rec_lag)
            d = dt.datetime(d.year, d.month, 1)
            self.receipts[d] += recs

    def add(self, id, amnt):
        self.ids.append(id)
        self.amnts.append(amnt)
        self.__gen_cfs__(id, amnt)

    def get_cashflows(self, d):
        dates = sorted(set(self.payments.keys()).union(set(self.receipts.keys())))
        dates = [dat for dat in dates if dat <= d]
        if dates == []:
            df1 = pd.DataFrame({'payments': 0, 'receipts': 0, 'net': 0, 'PV': 0}, index=[d])
        else:
            s1 = pd.Series([self.n_loans[dat] for dat in dates], index=dates, name='n_loans')
            s2 = pd.Series([self.payments[dat] for dat in dates], index=dates, name='payments')
            s3 = pd.Series([self.receipts[dat] for dat in dates], index=dates, name='receipts')
            s4 = pd.Series([self.PVs[dat] for dat in dates], index=dates, name='PVs')
            df1 = pd.concat([s1, s2, s3, s4], axis=1)
            df1['net'] = df1['payments'] + df1['receipts']
            df1 = df1[['n_loans', 'payments', 'receipts', 'net', 'PVs']]
        return(df1)

    def spent(self, d):
        # returns total net cashflows for the investor to date d
        df2 = self.get_cashflows(d)
        return(df2['net'].sum())

    def drawdown(self, d):
        # returns the worst of the cumulative net cashflows for the investor up to date d
        df3 = self.get_cashflows(d)
        df3['cum_sum'] = df3['net'].cumsum()
        return (df3['cum_sum'].min())

    def FV(self, d):
        # future value of net cashflows to date d
        df4 = self.get_cashflows(d)
        return(self.__yc__.FV(df4.index, df4['net']))

    def IRR(self, d):
        # IRR of net cashflows to date d
        df5 = self.get_cashflows(d)
        r = npf.irr(df5['net'])
        return(2 * ((1 + r)**6 - 1))