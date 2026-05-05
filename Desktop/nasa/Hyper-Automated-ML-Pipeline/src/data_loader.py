import pandas as pd

def load_dataset(path):
    df = pd.read_csv(
        path,
        sep=r'\s+',
        header=None
    )
    df = df.dropna(axis=1)
    columns = ["engine_id", "cycle", "op1", "op2", "op3"]
    columns += [f"sensor_{i}" for i in range(1, 22)]
    df.columns = columns
    return df

