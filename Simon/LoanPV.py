# Populates columns of RF_rate ('risk free rate') based on loan start and last payment date
# and PV of cash flows
# This file takes about 5 minutes to run

import pandas as pd

accept = pd.read_pickle('../derivedData/train.pkl')
# change int32 cols to datetime
accept['issue_d'] = pd.to_datetime(accept['issue_d'].astype(str), format='%Y%m%d')
accept['last_pymnt_d'] = pd.to_datetime(accept['last_pymnt_d'].astype(str), format='%Y%m%d')

rates = pd.read_csv('../macroData/rates.csv', index_col=0)
rates.index = pd.to_datetime(rates.index)

def PV(start, stop, loan, paid, recovered, rate=0, rec_lag=9):
    '''
    Returns PV at start date of cash flows
    Assumes ('paid' - 'recoveries') is equally distributed between start and stop
    start and stop should be datetime/Timestamp objects
    paid and recovered are USD, rate is annual % with semi-annual compounding
    and rec_lag is time in months after stop to receipt of recovery.
    BTW - recoveries are 1.6% of all payments, so let's not go nuts here.
    '''
    if rate == 0:
        PV = paid
    else:
        days = (stop - start).days
        months = int(days * 12 / 365.25)
        if (months == 0) & (rec_lag == 0):
            PV = paid
        else:
            # Convert s.a. rate to monthly (total overkill here)
            rate = 12 * ((1 + rate / 2) ** (1 / 6) - 1)
            s = 1 / (1 + rate / 12)
            if months == 0:
                PV = paid - recovered
            else:
                mthly_pymnt = (paid - recovered) / months
                PV = mthly_pymnt * (s - s**(months+1)) / (1 - s)
            PV = PV + recovered * s**(months + rec_lag)
    return(PV - loan)

def get_rate(df, start, stop):
    # return an appoximate risk free rate as of start for term (stop-start)
    days = (stop - start).days
    idx = df.index[df.index.get_loc(start, method='pad')]
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

print('Populating Rates')
df = accept[['issue_d', 'last_pymnt_d']]
accept['RF_rate'] = df.apply(lambda x: get_rate(rates, x[0], x[1]), axis=1)
print('-'*20)
print('Populating PVs')
df = accept[['issue_d', 'last_pymnt_d', 'funded_amnt', 'total_pymnt', 'recoveries', 'RF_rate']]
accept['PV'] = df.apply(lambda x: PV(x[0], x[1], x[2], x[3], x[4], x[5]), axis=1)
print('-'*20)

# return cols to int
accept['issue_d'] = accept['issue_d'].apply(lambda d: int(d.strftime('%Y%m%d')))
accept['last_pymnt_d'] = accept['last_pymnt_d'].apply(lambda d: int(d.strftime('%Y%m%d')))
accept.to_pickle('../derivedData/train.pkl')