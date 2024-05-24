import pandas as pd
import numpy as  np
import seaborn as sns
import matplotlib.pyplot as plt

import catboost as catt
import lightgbm as lgb
import xgboost as xgb

from tqdm import tqdm

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


train = pd.read_csv("../../../ctr/data/train_ctr.csv")
sub = pd.read_csv("../../../ctr/data/subs_ctr.csv")



def remove_nans(data, thresh):
    def nans_rate(data, col):
        return data[col].isna().sum() / data.shape[0]

    for col in data.columns:
        if nans_rate(data, col) >= thresh:
            data.drop(col, axis=1, inplace=True)

    return data

train = remove_nans(train, 0.7)

train.drop(columns = ['currency'], inplace=True)
train.sort_values(by = ['ID', 'date'], inplace=True)

def get_date(value):
    parts = value.split('_')
    date = '_'.join(parts[2:])
    return date

def get_id(value):
    parts = value.split('_')
    id_ = '_'.join(parts[:2])
    return id_

def week_of_month(date_val):
    first_day = date_val.replace(day=1)
    day_of_month = date_val.day
    adjusted_dom = day_of_month + first_day.weekday()
    return int(np.ceil(adjusted_dom/7))



def build_datasets(train, sub, agent_id):
    min_date_sub_str = sub[sub['ID'].str.contains(agent_id)]['ID'].str.split('_').str[-3:].min()
    min_date_sub = pd.to_datetime('_'.join(min_date_sub_str), format='%Y_%m_%d')
    train_subset = train[train['ID'] == agent_id]
    max_date_train = pd.to_datetime(train_subset['date']).max()
    missing_dates = pd.date_range(start=max_date_train + pd.Timedelta(days=1), end=min_date_sub - pd.Timedelta(days=1), freq='D')
    missing_rows = pd.DataFrame({'date': missing_dates})
    missing_rows['ID'] = agent_id
    train_with_missing = pd.concat([train_subset, missing_rows], ignore_index=True)
    train_with_missing['identifier'] = 'train'
    sub['identifier'] = 'test'
    train_with_missing['date'] = pd.to_datetime(train_with_missing['date'])
    date_range = pd.date_range(start=train_with_missing['date'].min(), end=train_with_missing['date'].max())
    all_dates = pd.DataFrame(date_range, columns=['date'])
    all_dates['ID'] = train_with_missing['ID'].iloc[0]
    merged_df = pd.merge(all_dates, train_with_missing, on=['date', 'ID'], how='left')
    merged_df['identifier'] = merged_df['identifier'].fillna('train')
    merged_df['unique_id'] = merged_df['ID'].astype(str) + "_" + merged_df['date'].dt.strftime('%Y-%m-%d')
    sub_ =sub[sub['ID'].str.contains(agent_id)]
    sub_['date'] = sub_['ID'].apply(get_date)
    sub_['ID'] = sub_['ID'].apply(get_id)
    sub_['date'] = pd.to_datetime(sub_['date'], format='%Y_%m_%d')
    concatenated_df = pd.concat([merged_df, sub_], ignore_index=True)

    concatenated_df['date'] = pd.to_datetime(concatenated_df['date'])
    # Extract features:
    concatenated_df['day'] = concatenated_df['date'].dt.day  # Day of the month
    concatenated_df['year'] = concatenated_df['date'].dt.year  # Day of the month
    concatenated_df['week'] = concatenated_df['date'].dt.isocalendar().week  # Week of the year
    concatenated_df['month'] = concatenated_df['date'].dt.month  # Month of the year
    concatenated_df['is_weekend'] = concatenated_df['date'].dt.dayofweek >= 5  # True for Sat/Sun
    concatenated_df['day_of_week'] = concatenated_df['date'].dt.dayofweek
    concatenated_df['week_of_month'] = concatenated_df['date'].apply(week_of_month)
    concatenated_df['is_month_start'] = concatenated_df.date.dt.is_month_start.astype(int)
    concatenated_df['is_month_end'] = concatenated_df.date.dt.is_month_end.astype(int)
    concatenated_df['quarter_of_year'] = concatenated_df.date.dt.quarter
    concatenated_df['day_of_year'] = concatenated_df.date.dt.dayofyear


    # Aghgregate the data by date
    df_agg = concatenated_df.groupby('date').agg({
        'impressions': 'sum',
        'clicks': 'sum',
        'ad_type': 'first',
        'ID': 'first',
        'day': 'first',
        'year': 'first',
        'week': 'first',
        'month': 'first',
        'is_weekend': 'first',
        'week_of_month': 'first',
        'quarter_of_year': 'first',
        'day_of_year': 'first',
        'is_month_start': 'first',
        'is_month_end': 'first',
        'identifier': 'first'

    }).reset_index()

    # Group the DataFrame by week
    df_month = concatenated_df.groupby(['month']).agg({
        'impressions': ['mean', 'std', 'min', 'max', 'skew'],
        'cost': ['mean', 'std', 'min', 'max', 'skew'],

    }).reset_index()
    df_month.columns = ['month'] + [f"month_{col}_{stat}" for col, stat in df_month.columns[1:]]
    df_weekly = concatenated_df.groupby('week').agg({
        'impressions': ['mean', 'std', 'min', 'max', 'skew'],
        'cost': ['mean', 'std', 'min', 'max', 'skew'],

    }).reset_index()
    df_weekly.columns = ['week'] + [f"weekly_{col}_{stat}" for col, stat in df_weekly.columns[1:]]

    df_agg = pd.merge(df_agg, df_month, on='month', how='left')
    df_agg = pd.merge(df_agg, df_weekly, on='week', how='left')

    le_cols = ['ad_type']
    stats_cols = [col for col in df_agg.columns if any(substring in col for substring in ['mean', 'std', 'min', 'max', 'skew'])]

    lag_cols = [col for col in df_agg.columns if col not in stats_cols + ['date', 'ID','ad_type', 'unique_id', 'is_weekend', 'week_of_month', 'quarter_of_year', 'day_of_year',
       'is_month_start', 'is_month_end', 'day', 'year', 'week', 'month', 'identifier' ]]#and not any(suffix in col for suffix in ['mean', 'std', 'min', 'max', 'skew']
    date_cols = [ 'is_weekend', 'week_of_month', 'quarter_of_year', 'day_of_year',
           'is_month_start', 'is_month_end', 'day', 'year', 'week', 'month' ]
    for col in le_cols:
        df_agg[col] = le.fit_transform(df_agg[col])

    for col in lag_cols:
        for shift in range(15, 23):
            df_agg[f"lagged_{col}_{shift}"] = df_agg[col].shift(shift)

    for col in stats_cols:
        for shift in [15, 23, 30]:
            df_agg[f"lagged_{col}_{shift}"] = df_agg[col].shift(shift)


    for col in lag_cols:
        for shift in range(16, 23):
            lagged_col = f"lagged_{col}_{shift}"
            df_agg[f"diff_{col}_lagged_{shift}"] = df_agg[f"lagged_{col}_15"] - df_agg[lagged_col]

    for col in stats_cols:
        for shift in [23, 30]:
            lagged_col = f"lagged_{col}_{shift}"
            df_agg[f"diff_{col}_lagged_{shift}"] = df_agg[f"lagged_{col}_15"] - df_agg[lagged_col]

    independent_features =  [col for col in df_agg.columns if 'lagged' in col] + date_cols + [col for col in df_agg.columns if 'roll_mean' in col]
    df_agg = df_agg[['clicks', 'identifier', 'date', 'ID'] + independent_features + stats_cols]
    df_agg['unique_id'] = df_agg['ID'].astype(str) + "_" + df_agg['date'].dt.strftime('%Y-%m-%d')

    df_agg_train = df_agg[df_agg['identifier']=='train']

    n = len(df_agg_train)
    counter = 0
    selected_rows = []

    for i in range(n - 1, -1, -1):
        counter += 1
        if counter == 8 or counter == 15:
            selected_rows.append(df_agg_train.iloc[i])
        if counter == 15:
            counter = 0

    train_df = pd.DataFrame(selected_rows)
    test_df = df_agg[df_agg['identifier']=='test']
    return train_df, test_df


all_train_sets = []
all_test_sets = []
for agent_id in tqdm(train['ID'].unique()):
    train_set, test_set = build_datasets(train, sub, agent_id)
    all_train_sets.append(train_set)
    all_test_sets.append(test_set)
print(len(all_train_sets), len(all_test_sets))


final_train_df = pd.concat(all_train_sets)
final_test_df = pd.concat(all_test_sets)
final_train_df['ID'] = le.fit_transform(final_train_df['ID'])
final_test_df['ID'] = le.transform(final_test_df['ID'])
print(final_train_df.shape)
print(final_train_df.head()) 
print(final_test_df.shape)
print(final_test_df.head())


stats_cols = [col for col in final_train_df.columns if any(substring in col for substring in ['mean', 'std', 'min', 'max', 'skew'])]
date_cols = [ 'is_weekend', 'week_of_month', 'quarter_of_year', 'day_of_year',
       'is_month_start', 'is_month_end', 'day', 'year', 'week', 'month' ]
independent_features =  [col for col in final_train_df.columns if 'lagged' in col] + ['ID']+ date_cols + [col for col in stats_cols if 'lagged' not in col] + [col for col in final_train_df.columns if 'roll_mean' in col]


final_train_df = final_train_df.sort_values(by=['ID', 'date']).reset_index(drop=True)
from sklearn.model_selection import TimeSeriesSplit
n_splits = 4
tscv = TimeSeriesSplit(n_splits=n_splits)


fold_preds = []
mse_scores = []
df = final_train_df.copy()
X = df[independent_features]
y = df['clicks']

# Perform time series cross-validation
for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    model = catt.CatBoostRegressor(random_state=42, n_estimators = 500)
    model.fit(X_train, y_train, eval_set = (X_test, y_test), verbose = 1000)


    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    mse_scores.append(mse)

    print(f"rmse: {mse}")
    test_preds = model.predict(final_test_df[independent_features])
    fold_preds.append(test_preds)


avg_mse = np.mean(mse_scores)
print("Average RSE:", avg_mse)


feature_importance_df = pd.DataFrame(model.feature_importances_, columns=['importance'])
feature_importance_df['feature'] = X.columns



preds = np.mean(fold_preds, axis = 0)
final_test_df['preds'] = preds
sub = final_test_df[['unique_id', 'preds']]
sub['unique_id'] = sub['unique_id'].str.replace("-", "_")
sub.to_csv('12.27_sub.csv', index=False)

model = catt.CatBoostRegressor(random_state = 42, n_estimators = 500)
model.fit(X, y, verbose = False)
test_preds = model.predict(final_test_df[independent_features])
final_test_df['preds'] = test_preds
sub = final_test_df[['unique_id', 'preds']]
sub['unique_id'] = sub['unique_id'].str.replace("-", "_")
sub.to_csv('full_model_sub.csv', index=False)


