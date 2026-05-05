from sklearn.preprocessing import StandardScaler

def scale_data(df):

    features = [col for col in df.columns if "sensor" in col]

    scaler = StandardScaler()

    df[features] = scaler.fit_transform(df[features])

    return df