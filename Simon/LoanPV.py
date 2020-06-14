# Need to redo:  first figure out original pymnt schedule from term and rate
# Estimate total payment based on last payment date
# add a balancing item
# add recoveries
# PV loan

import pandas as pd
import datetime as dt

accept = pd.read_pickle('../derivedData/train.pkl')
accept.set_index('id', inplace = True)

def PV(start, stop, loan, paid, recovered, rate=0, rec_lag=9):
    '''
    Returns PV at start date of cash flows
    Assumes 'paid' is equally distributed between start and stop
    start and stop should be datetime/Timestamp objects
    paid and recovered are USD, rate is annual % with semi-annual compounding
    and rec_lag is time in months after stop to receipt of recovery
    '''
    if rate == 0:
        PV = paid + recovered
    else:
        days = (stop - start).days
        months = int(days * 12 / 365.25)
        if (months == 0) & (rec_lag == 0):
            PV = paid + recovered
        else:
            # Convert s.a. rate to monthly (total overkill here)
            rate = 12 * ((1 + rate / 2) ** (1 / 6) - 1)
            s = 1 / (1 + rate / 12)
            if months == 0:
                PV = paid
            else:
                mthly_pymnt = paid / months
                PV = mthly_pymnt * (s - s**(months+1)) / (1 - s)
            PV = PV + recovered * s**(months + rec_lag)
    return(PV - loan)

df = accept[['issue_d', 'last_pymnt_d', 'funded_amnt', 'total_pymnt', 'recoveries']]
accept['PV'] = df.apply(lambda x: PV(x[0], x[1], x[2], x[3], x[4]), axis=1)

accept.to_pickle('../derivedData/train.pkl')