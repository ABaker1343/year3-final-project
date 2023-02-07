import pandas

def load_time_series(path : str):
    df = pandas.read_csv(path)
    return df

def with_unix(df : pandas.DataFrame, date_field="Date") -> pandas.DataFrame:
    unix_times = [(pandas.to_datetime(row[date_field]) - pandas.Timestamp("1970-01-01")) // pandas.Timedelta('1s') for _, row in df.iterrows()]
    df['Unix'] = unix_times
    return df
