def create_rul(df):
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["max_cycle"] = df["engine_id"].map(max_cycle)
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop("max_cycle", axis=1, inplace=True)
    return df

