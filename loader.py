import pandas

def load_time_series(path : str):
    df = pandas.read_csv(path)
    return df

def with_unix(df : pandas.DataFrame, date_field="Date") -> pandas.DataFrame:
    #unix_times = [(pandas.to_datetime(row[date_field]) - pandas.Timestamp("1970-01-01")) for _, row in df.iterrows()]
    unix_times = pandas.to_datetime(df[date_field]).view('int64')
    df['Unix'] = unix_times
    return df

def split_dates(df : pandas.DataFrame, date_field="Date") -> pandas.DataFrame:
    days = []
    months = []
    years = []
    for _, row in df.iterrows():
        dt = pandas.to_datetime(row[date_field])
        days.append(dt.day)
        months.append(dt.month)
        years.append(dt.year)
    df["Day"] = days
    df["Month"] = months
    df["Year"] = years
    return df
        
