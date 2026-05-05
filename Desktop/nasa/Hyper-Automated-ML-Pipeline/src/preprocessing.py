from sklearn.preprocessing import StandardScaler

scaler_dict = {}

def scale_data(df, scaler=None, fit=False):
    features = [col for col in df.columns if "sensor" in col]
    scaler = StandardScaler() if fit else scaler
    df[features] = scaler.fit_transform(df[features]) if fit else scaler.transform(df[features])
    return df, scaler


def get_scaler(df):
    features = [col for col in df.columns if "sensor" in col]
    if features:
        scaler = StandardScaler()
        scaler.fit(df[features])
        scaler_dict['scaler'] = scaler
    return scaler_dict['scaler']

