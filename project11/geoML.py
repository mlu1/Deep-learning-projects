import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
import geohash as gh

train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')


'''
for col in test_data.columns:
    if col not in ['ID','ID_Zindi','Date']:
        train_data[col] = train_data[col].astype(float).replace([np.inf, -np.inf],np.nan)
        test_data[col] = test_data[col].astype(float).replace([np.inf, -np.inf],np.nan)
'''        

def fillna(df, target, strategy='mean'):
    # Identify columns with NaN values, excluding the target column
    nan_columns = df.columns[df.isna().any()].tolist()
    if target in nan_columns:
        nan_columns.remove(target)
    
    # Fill NaN values based on the specified strategy for each column
    for col in nan_columns:
        if strategy == 'mean':
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == 'median':
            df[col] = df[col].fillna(df[col].median())
        elif strategy == 'mode':
            df[col] = df[col].fillna(df[col].mode()[0])  # Use the first mode value if multiple modes exist
        else:
            raise ValueError("Unsupported strategy. Choose 'mean', 'median', or 'mode'.")
    
    return df

train = fillna(train_data, target='target', strategy='mean')
test = fillna(test_data, target='target', strategy='mean')



#train = train_data.fillna(0)
#test = test_data.fillna(0)




le_cols = ['ID','geohash']



def count(df, columns):
    for col in columns:
        df['count_' + col] = df[col].map(df[col].value_counts())
    return df


def encoder(df, cols):
    label_encoder = LabelEncoder()
    for col in cols:
        df[col] = label_encoder.fit_transform(df[col])
    return df


def week_of_month(date_val):
    first_day = date_val.replace(day=1)
    day_of_month = date_val.day
    adjusted_dom = day_of_month + first_day.weekday()
    return int(np.ceil(adjusted_dom/7))

def extract_date_info(dataframe):
    dataframe['Date'] = pd.to_datetime(dataframe['Date'])
    dataframe['DayOfWeek'] = dataframe['Date'].dt.dayofweek
    dataframe['Month'] = dataframe['Date'].dt.month
    dataframe['Year'] = dataframe['Date'].dt.year
    dataframe['Week'] = dataframe['Date'].dt.isocalendar().week
    dataframe['day'] = dataframe['Date'].dt.day  # Day of the month

    dataframe['is_weekend'] = dataframe['Date'].dt.dayofweek >= 5  # True for Sat/Sun
    dataframe['day_of_week'] = dataframe['Date'].dt.dayofweek
    
    dataframe['week_of_month'] = dataframe['Date'].apply(week_of_month)

    
    dataframe['is_month_end'] = dataframe.Date.dt.is_month_end.astype(np.int8)
    dataframe['monday'] = dataframe.Date.dt.weekday.eq(0).astype(np.uint8)
    dataframe['tuesday'] = dataframe.Date.dt.weekday.eq(1).astype(np.uint8)
    dataframe['wednesday'] = dataframe.Date.dt.weekday.eq(2).astype(np.uint8)
    dataframe['thursday'] = dataframe.Date.dt.weekday.eq(3).astype(np.uint8)
    dataframe['friday'] = dataframe.Date.dt.weekday.eq(4).astype(np.uint8)
    dataframe['saturday'] = dataframe.Date.dt.weekday.eq(5).astype(np.uint8)
    dataframe['sunday'] = dataframe.Date.dt.weekday.eq(6).astype(np.uint8)


    dataframe['is_month_start'] = dataframe['Date'].dt.is_month_start.astype(int)
    dataframe['quarter_of_year'] = dataframe['Date'].dt.quarter
    dataframe['day_of_year'] = dataframe['Date'].dt.dayofyear
    
    dataframe['is_year_end'] = dataframe['Date'].dt.is_year_end
    dataframe['is_year_start'] = dataframe['Date'].dt.is_year_start

    dataframe['sin_dayofweek'] = np.sin(2 * np.pi * dataframe['DayOfWeek'] / 7)
    dataframe['cos_dayofweek'] = np.cos(2 * np.pi * dataframe['DayOfWeek'] / 7)
    dataframe['sin_week'] = np.sin(2 * np.pi * dataframe['Week'] / 52)
    dataframe['cos_week'] = np.cos(2 * np.pi * dataframe['Week'] / 52)
    dataframe['sin_month'] = np.sin(2 * np.pi * dataframe['Month'] / 12)
    dataframe['cos_month'] = np.cos(2 * np.pi * dataframe['Month'] / 12)
        

    dataframe.drop(['Date'], axis=1, inplace=True)
    return dataframe


train = extract_date_info(train)
test = extract_date_info(test)




def geohash(df, CFG):
    # Generate geohashes based on latitude, longitude, and specified precision
    df['geohash'] = df.apply(lambda row: gh.encode(row['LAT'], row['LON'], precision=CFG.data_preparation.precision), axis=1)
    return df

# Example DataFrame


# Example CFG with precision
class Config:
    class DataPreparation:
        precision = 5
    data_preparation = DataPreparation()

CFG = Config()


train = geohash(train, CFG)
test = geohash(test, CFG)


print(train['geohash'])


train = encoder(train, cols=le_cols)
test = encoder(test, cols=le_cols)

train = count(train, columns=le_cols)
test = count(test, columns=le_cols)



clus_comp: int = 23
random_state:int = 45
clus_cols = ['NO2_strat','NO2_total','NO2_trop'] 

scaler = StandardScaler()
scaler.fit(train[clus_cols])

x_scaled_tr = pd.DataFrame(scaler.transform(train[clus_cols]), index = train.index, columns = clus_cols)
x_scaled_ts = pd.DataFrame(scaler.transform(test[clus_cols]), index = test.index, columns = clus_cols)

  # cluster
model = GaussianMixture(n_components = clus_comp, random_state = random_state)
model.fit(x_scaled_tr)

train['clusters'] = model.predict(x_scaled_tr)
test['clusters'] = model.predict(x_scaled_ts)



lag_cols = ['Precipitation','LST','AAI','CloudFraction','NO2_strat','NO2_total','NO2_trop','TropopausePressure']



print(train[lag_cols].head(10))



scaler = StandardScaler()
scaler.fit(train[clus_cols])

x_scaled_tr = pd.DataFrame(scaler.transform(train[clus_cols]), index = train.index, columns = clus_cols)
x_scaled_ts = pd.DataFrame(scaler.transform(test[clus_cols]), index = test.index, columns = clus_cols)

pca = PCA(n_components = 1, random_state = random_state)
pca.fit(x_scaled_tr)

pca_data_tr = pca.transform(x_scaled_tr)
pca_data_ts = pca.transform(x_scaled_ts)

for i in range(len(pca_data_tr.T)):
    train[f'pc_{i+1}'] = pca_data_tr[:, i]
    test[f'pc_{i+1}'] = pca_data_ts[:, i]      


for col in lag_cols:
    for shift in range(15, 23):
        train[f"lagged_{col}_{shift}"] = train[col].shift(shift)
        test[f"lagged_{col}_{shift}"] = test[col].shift(shift)


for col in lag_cols:
    for shift in range(16, 23):
        lagged_col = f"lagged_{col}_{shift}"
        train[f"diff_{col}_lagged_{shift}"] = train[f"lagged_{col}_15"] - train[lagged_col]
        test[f"diff_{col}_lagged_{shift}"] = test[f"lagged_{col}_15"] - test[lagged_col]
        
    
train['vpd_div_vap'] = (train['NO2_strat'] / train['NO2_total']).fillna(0)
test['vpd_div_vap'] = (test['NO2_strat'] / test['NO2_total']).fillna(0)
train['ele_mul_vs'] = (train['NO2_strat'] * train['NO2_trop']).fillna(0)
train['srad_mul_pet'] = (train['Precipitation'] * train['AAI']).fillna(0)
train['def_div_aet'] = (train['AAI'] * train['LST']).fillna(0)
train['srad_mul_pet'] = (train['TropopausePressure'] * train['CloudFraction']).fillna(0)
train['no_total'] = (abs(train['NO2_strat'] - train['NO2_trop'])).fillna(0)



test['ele_mul_vs'] = (test['NO2_strat'] * test['NO2_trop']).fillna(0)
test['srad_mul_pet'] = (test['Precipitation'] * test['AAI']).fillna(0)
test['def_div_aet'] = (test['AAI'] * test['LST']).fillna(0)
test['srad_mul_pet'] = (test['TropopausePressure'] * test['CloudFraction']).fillna(0)
test['no_total'] = (abs(test['NO2_strat'] - test['NO2_trop'])).fillna(0)



train['location'] = train['LAT'].astype('str') + '_' + train['LON'].astype('str')
test['location'] = test['LAT'].astype('str') + '_' + test['LON'].astype('str')


nan_cols=['Precipitation','CloudFraction','LST','AAI','NO2_total']

for col in nan_cols:
        train[col].fillna(train[["location","ID", col]].groupby(["ID","location"]).shift(periods=0).fillna(method='bfill', limit=1).fillna(method='bfill', limit=1)[col], inplace=True)
        test[col].fillna(test[["location","ID", col]].groupby(["ID","location"]).shift(periods=0).fillna(method='bfill', limit=1).fillna(method='bfill', limit=1)[col], inplace=True)



#train['def_div_aet'] = (train['LST'] / train['climate_aet']).fillna(0)
#train['def_div_aet'] = np.where(np.isinf(train['def_div_aet']), 0, train['def_div_aet'])


def group_stats(columns: list, df: pd.DataFrame, hide_p_bar: bool = False) -> pd.DataFrame:
    """
    Calculate statistics for specified columns in a DataFrame.

    Parameters:
    - columns (list): List of column names to calculate statistics on.
    - df (pd.DataFrame): Input DataFrame containing the data.
    - hide_p_bar (bool): Whether to hide the progress bar. Defaults to False.

    Returns:
    - pd.DataFrame: DataFrame with additional columns for calculated statistics.
    """
    stats_df = df.copy()

    group_df = df[columns].dropna(axis=1, how='all')  # Drop columns with all NaNs

    if not group_df.empty:
        stats_df["std"] = group_df.std(axis=1)
        stats_df["mean"] = group_df.mean(axis=1)
        stats_df["sum"] = group_df.sum(axis=1)
        stats_df["prod"] = group_df.prod(axis=1)

        if len(columns) > 2:
            stats_df["min"] = group_df.min(axis=1)
            stats_df["max"] = group_df.max(axis=1)
            stats_df["median"] = group_df.median(axis=1)

    return stats_df

train = group_stats(clus_cols, train)
test = group_stats(clus_cols, test)



df_weekly = train.groupby('Week').agg({
        'Precipitation': ['mean', 'std', 'min', 'max', 'skew'],
        'no_total': ['mean', 'std', 'min', 'max', 'skew'],
        'LST':['mean', 'std', 'min', 'max', 'skew'],
        'TropopausePressure': ['mean', 'std', 'min', 'max', 'skew'],
        'CloudFraction': ['mean', 'std', 'min', 'max', 'skew'], 
        'AAI': ['mean', 'std', 'min', 'max', 'skew'], 
 
    }).reset_index()


df_weekly_test = test.groupby('Week').agg({
        'Precipitation': ['mean', 'std', 'min', 'max', 'skew'],
        'no_total': ['mean', 'std', 'min', 'max', 'skew'],
        'LST':['mean', 'std', 'min', 'max', 'skew'],
        'TropopausePressure': ['mean', 'std', 'min', 'max', 'skew'],
        'CloudFraction': ['mean', 'std', 'min', 'max', 'skew'], 
        'AAI': ['mean', 'std', 'min', 'max', 'skew'], 
        
        }).reset_index()



df_weekly.columns = ['Week'] + [f"weekly_{col}_{stat}" for col, stat in df_weekly.columns[1:]]
df_weekly_test.columns = ['Week'] + [f"weekly_{col}_{stat}" for col, stat in df_weekly_test.columns[1:]]



train = pd.merge(train, df_weekly, on='Week', how='left')
test = pd.merge(test, df_weekly_test, on='Week', how='left')




def rolling(cols: list, df: pd.DataFrame, roll_d: dict = None, overall: bool = True,
            w_z: int = 3, w_med: int = 3, w_std: int = 3, w_skw: int = 3,
            w_kurt: int = 3, w_var: int = 3, w_cov: int = 3, w_sem: int = 3,
            hide_p_bar: bool = False):
    """
    Calculate rolling statistics for specified columns in a DataFrame, either overall or area-specific.

    Parameters:
    - cols (list): List of column names to calculate rolling statistics on.
    - df (pd.DataFrame): Input DataFrame containing the data.
    - roll_d (dict): Dictionary specifying custom aggregation functions and window sizes for columns.
    - overall (bool): If True, calculate overall rolling stats; if False, calculate area-specific.
    - w_z, w_med, w_std, w_skw, w_kurt, w_var, w_cov, w_sem (int): Window sizes for rolling calculations.
    - hide_p_bar (bool): Whether to hide the progress bar. Defaults to False.
    """
    desc = 'Calculating Rolling Stats'

    if roll_d:
        for c in (cols):
            agg = roll_d[c]['agg']
            w = roll_d[c]['w']
            df[f'r_{c}_{agg}'] = df[c].rolling(window=w).agg(agg).fillna(0)
    else:
        if overall:
            for c in (cols):
                df[f'r_z_{c}_o'] = ((df[c] - df[c].rolling(w_z).mean()) / df[c].rolling(w_z).std()).fillna(0)
                df[f'r_med_{c}_o'] = df[c].rolling(w_med).median().fillna(0)
                df[f'r_std_{c}_o'] = df[c].rolling(w_std).std().fillna(0)
                df[f'r_skew_{c}_o'] = df[c].rolling(w_skw).skew().fillna(0)
                df[f'r_kurt_{c}_o'] = df[c].rolling(w_kurt).kurt().fillna(0)
                df[f'r_var_{c}_o'] = df[c].rolling(w_var).var().fillna(0)
                df[f'r_cov_{c}_o'] = df[c].rolling(w_cov).cov().fillna(0)
                df[f'r_sem_{c}_o'] = df[c].rolling(w_sem).sem().fillna(0)
        else:
            g = df.groupby('Id')
            for c in tqdm(cols):
                df[f'r_z_{c}_a'] = ((df[c] - g[c].transform(lambda x: x.rolling(w_z).mean())) /
                                     g[c].transform(lambda x: x.rolling(w_z).std())).fillna(0)
                df[f'r_med_{c}_a'] = g[c].transform(lambda x: x.rolling(w_med).median()).fillna(0)
                df[f'r_std_{c}_a'] = g[c].transform(lambda x: x.rolling(w_std).std()).fillna(0)
                df[f'r_skew_{c}_a'] = g[c].transform(lambda x: x.rolling(w_skw).skew()).fillna(0)
                df[f'r_kurt_{c}_a'] = g[c].transform(lambda x: x.rolling(w_kurt).kurt()).fillna(0)
                df[f'r_var_{c}_a'] = g[c].transform(lambda x: x.rolling(w_var).var()).fillna(0)
                df[f'r_cov_{c}_a'] = g[c].transform(lambda x: x.rolling(w_cov).cov()).fillna(0)
                df[f'r_sem_{c}_a'] = g[c].transform(lambda x: x.rolling(w_sem).sem()).fillna(0)

    return df



import folium
train = rolling(clus_cols, train, overall=True)
test = rolling(clus_cols, test, overall=True)


my_map = folium.Map(
    location=(train['LAT'].mean(), train['LON'].mean()),
    zoom_start=7,
)


unique_train_locations = train.groupby(['LAT', 'LON'])['GT_NO2'].mean().reset_index()

layer_train_map = folium.FeatureGroup(name='Train Locations', show=False)
for index, row in unique_train_locations.iterrows():
    folium.Marker(
        location=[row['LAT'], row['LON']],
        icon=folium.Icon(color='green', icon='home'),
        popup=f'Mean GT_NO2 level: {row["GT_NO2"]:.2f}',
    ).add_to(layer_train_map)

layer_train_map.add_to(my_map)


unique_test_locations = test[['LAT', 'LON']].drop_duplicates()

layer_test_map = folium.FeatureGroup(name='Test Locations', show=False)
for index, row in unique_test_locations.iterrows():
    folium.Marker(
        location=[row['LAT'], row['LON']],
        icon=folium.Icon(color='red', icon='home'),
    ).add_to(layer_test_map)
    
layer_test_map.add_to(my_map)
folium.LayerControl().add_to(my_map)
my_map.save('my_map.html')

test_ids = test['ID_Zindi']
train.reset_index(drop=True, inplace=True)
train.drop(columns=['ID','ID_Zindi','location'], axis=1, inplace=True)
test.drop(columns=['ID','ID_Zindi','location'], axis=1, inplace=True)


from scipy.stats import zscore
detect_outliers = zscore(train['GT_NO2'])

quantiles = pd.DataFrame(list(zip(np.linspace(0.98, 1, 21), [np.quantile(detect_outliers, el) for el in np.linspace(0.98, 1, 21)], [np.quantile(train['GT_NO2'], el) for el in np.linspace(0.98, 1, 21)])), columns=['quantile', 'zscore', 'GT_NO2'])
quantiles


def drop_low_correlated_columns_to_pm2_5():
    corr = train.corr(numeric_only=True)['GT_NO2'].to_frame()
    return corr[(corr['GT_NO2'] < 0.01) & (corr['GT_NO2'] > -0.01)].index.to_numpy()


to_drop = drop_low_correlated_columns_to_pm2_5()

print(to_drop)

train, test = train.drop(columns=to_drop, axis=1), test.drop(columns=to_drop, axis=1)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X, y = train.drop(columns=['GT_NO2'], axis=1), train['GT_NO2']


scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=scaler.feature_names_in_)
test = pd.DataFrame(scaler.transform(test), columns=scaler.feature_names_in_)

import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, GroupKFold


def define_lightgbm_model(trial):
    params = {
        'objective': 'root_mean_squared_error',
        'boosting_type': 'gbdt',
        'max_bin': trial.suggest_int('max_bin', 10, 200),
        'num_leaves': trial.suggest_int('num_leaves', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 9e-2, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 200, 700),
        'tree_learner': 'voting',
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0, log=True),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.8, log=True),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 50, 250),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 1, log=True),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        'random_state': 4,
        'verbosity': -1,
    }
    return lgb.LGBMRegressor(**params)

def objective_lightgbm(trial):
    model = define_lightgbm_model(trial)
    gkf = GroupKFold(n_splits=X['DayOfWeek'].nunique())
    scores = cross_val_score(model, X, y, groups=X['DayOfWeek'], cv=gkf, scoring='neg_root_mean_squared_error')
    return scores.mean() * (-1)
study_lightgbm = optuna.create_study(direction='minimize', study_name='AirQualityWithLightGBM', sampler=optuna.samplers.TPESampler())
study_lightgbm.optimize(objective_lightgbm, n_trials=200)
lgb_model = define_lightgbm_model(study_lightgbm.best_trial)
lgb_model.fit(X, y)
lightgbm_params = ['max_bin', 'num_leaves', 'max_depth', 'learning_rate', 'n_estimators', 'bagging_fraction', 'colsample_bytree', 'min_data_in_leaf']

from sklearn.model_selection import LearningCurveDisplay


LearningCurveDisplay.from_estimator(lgb_model, X, y, cv=10, random_state=4, scoring='neg_root_mean_squared_error')

study_lightgbm.best_params
def save_to_csv(y_pred, save_as):
    if 'result' not in os.listdir(os.getcwd()):
        os.mkdir('result')
    final_df = pd.concat([test_ids, pd.DataFrame.from_dict({'GT_NO2': y_pred})], axis=1)
    final_df.to_csv(os.path.join('result', save_as), index=False)
    
save_to_csv(lgb_model.predict(test), '10.csv')


