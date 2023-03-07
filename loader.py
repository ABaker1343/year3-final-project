import pandas

def load_time_series(file, fields, num_prev_days=0, prev_fields="", date_field="Date"):
    df = pandas.read_csv(file)
    df = with_unix(df, date_field)
    df = split_dates(df, date_field)
    df = with_days(df, date_field)

    if num_prev_days > 0:
        df = add_prev_days(df, prev_fields, num_prev_days)

    df = df[fields]

    return df

def with_unix(df : pandas.DataFrame, date_field="Date") -> pandas.DataFrame:
    #unix_times = [(pandas.to_datetime(row[date_field]) - pandas.Timestamp("1970-01-01")) for _, row in df.iterrows()]
    unix_times = pandas.to_datetime(df[date_field]).view('int64')
    df['Unix'] = unix_times
    return df

def with_days(df : pandas.DataFrame, date_field="Date") -> pandas.DataFrame:
    days = []
    for i, row in df.iterrows():
        days.append(i)
    df["Day"] = days
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
        
def add_prev_days(df : pandas.DataFrame, field : str, num_days=5) -> pandas.DataFrame:
    for i in range(1, num_days + 1):
        df[field + str(i) + "Day"] = df[field].shift(periods=i)

    return df

def normalize_dataframe(df : pandas.DataFrame):
    return (df - df.min()) / (df.max() - df.min())

def regularize_dataframe(df : pandas.DataFrame):
    return (df - df.mean()) / df.std()
