from sklearn.preprocessing import StandardScaler

def normalize_arr(fit_arr, *arrs):
    sc = StandardScaler()
    fit_arr = sc.fit_transform(fit_arr)
    out=[]
    for arr in arrs:
        out.append(sc.transform(arr))
    return [fit_arr]+out