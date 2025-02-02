import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
import optuna
from sklearn.metrics import mean_squared_error
import geohash as gh
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# Load datasets
train = pd.read_csv("Train.csv")
test = pd.read_csv("Test.csv")
toilets = pd.read_csv("toilets.csv")
waste_management = pd.read_csv("waste_management.csv")
water_sources = pd.read_csv("water_sources.csv")

print(train.columns)

print(water_sources.head(10))
hospital_data = pd.concat([train, test])


def geohash(df, CFG):
    # Generate geohashes based on latitude, longitude, and specified precision
    df['geohash'] = df.apply(lambda row: gh.encode(row['Transformed_Latitude'], row['Transformed_Longitude'], precision=CFG.data_preparation.precision), axis=1)
    return df

# Example CFG with precision
class Config:
    class DataPreparation:
        precision = 5
    data_preparation = DataPreparation()

CFG = Config()

train = geohash(train, CFG)
test = geohash(test, CFG)

for df in [toilets, waste_management, water_sources]:
    df.drop(columns=['Year', 'Month'], inplace=True)


def rename_columns(df, prefix):
    for col in df.columns:
        if col not in ['Month_Year_lat_lon', 'lat_lon']:
            df.rename(columns={col: f"{prefix}_{col}"}, inplace=True)

rename_columns(toilets, "toilet")
rename_columns(waste_management, "waste")
rename_columns(water_sources, "water")

hospital_data['Total'].fillna(0, inplace=True)
water_sources.dropna(subset=['water_Transformed_Latitude'], inplace=True)

def find_nearest(hospital_df, location_df, lat_col, lon_col, id_col):
    # Create a cKDTree for efficient nearest neighbour search
    tree = cKDTree(location_df[[lat_col, lon_col]].values)
    nearest = {}
    # Loop through each hospital and find the nearest site in location_df
    for _, row in hospital_df.iterrows():
        _, idx = tree.query([row['Transformed_Latitude'], row['Transformed_Longitude']])
        nearest[row['ID']] = location_df.iloc[idx][id_col]
    return nearest

for df, prefix in [(toilets, 'toilet'), (waste_management, 'waste'), (water_sources, 'water')]:
    df[f"{prefix}_Month_Year_lat_lon"] = (
        df[f"{prefix}_Month_Year"] + '_' +
        df[f"{prefix}_Transformed_Latitude"].astype(str) + '_' +
        df[f"{prefix}_Transformed_Longitude"].astype(str)
    )


merged_data = hospital_data.copy()
datasets = [
    (toilets, 'toilet', 'toilet_Month_Year_lat_lon'),
    (waste_management, 'waste', 'waste_Month_Year_lat_lon'),
    (water_sources, 'water', 'water_Month_Year_lat_lon'),
]

for df, prefix, id_col in datasets:
    nearest = find_nearest(merged_data, df, f"{prefix}_Transformed_Latitude", f"{prefix}_Transformed_Longitude", id_col)
    nearest_df = pd.DataFrame(list(nearest.items()), columns=['ID', id_col])
    merged_data = merged_data.merge(nearest_df, on="ID").merge(df, on=id_col)


print(merged_data.columns)


categorical_cols = merged_data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if (col == 'ID'):
        continue
    else:
        le = LabelEncoder()
        merged_data[col] = le.fit_transform(merged_data[col])

for col in merged_data.columns:
    print(col)


print(len(merged_data))


water_cols =['water_10u','water_10v','water_2d','water_2t','water_evabs','water_evaow','water_evatc',
'water_evavt','water_albedo','water_lshf','water_lai_hv','water_lai_lv','water_pev','water_ro',
'water_src','water_skt','water_es','water_stl1','water_stl2','water_stl3',
'water_stl4','water_ssro','water_slhf','water_ssr','water_str','water_sp','water_sro'
,'water_sshf'
,'water_ssrd'
,'water_strd'
,'water_e'
,'water_tp'
,'water_swvl1'
,'water_swvl2'
,'water_swvl3'
,'water_swvl4'] 


toilet_cols = ['toilet_10u','toilet_10v','toilet_2d','toilet_2t','toilet_evabs','toilet_evaow','toilet_evatc'
,'toilet_evavt','toilet_albedo','toilet_lshf','toilet_lai_hv','toilet_pev','toilet_ro'
,'toilet_src','toilet_skt','toilet_es','toilet_stl1','toilet_stl2','toilet_stl3'
,'toilet_stl4','toilet_ssro','toilet_slhf','toilet_ssr','toilet_str','toilet_sp','toilet_sro','toilet_sshf','toilet_ssrd',
'toilet_strd','toilet_e','toilet_tp','toilet_swvl1','toilet_swvl2','toilet_swvl3','toilet_swvl4'
]

cat_cols = ['Location','Disease','Category_Health_Facility_UUID']

averages_toilet = merged_data[toilet_cols].mean()
sums_toilet = merged_data[toilet_cols].sum()

averages_water = merged_data[water_cols].mean()
sums_water = merged_data[water_cols].sum()


merged_data['averaged_water'] = averages_water 
merged_data['averaged_water'] = sums_water 

merged_data['averaged_toilt'] = averages_toilet 
merged_data['averaged_toilet'] = sums_toilet 



lender_agg = merged_data.groupby('Category_Health_Facility_UUID').agg({
    'toilet_10u': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_2d': ['mean', 'max', 'min', 'std'],
    'toilet_2t': ['mean', 'std', 'max', 'min'],
    'toilet_evabs': ['mean', 'sum', 'std'],
    'toilet_evaow': ['mean', 'max', 'min', 'std'],
    'toilet_lshf': ['mean', 'sum', 'std'],
    'toilet_lai_hv': ['mean', 'sum', 'std'],
    'toilet_lai_lv': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'], 
    'toilet_swvl4': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'],
    'toilet_ro': ['mean', 'sum', 'std'],
    'toilet_src':['mean', 'sum', 'std'],
    'toilet_skt': ['mean', 'sum', 'std'],
    'toilet_es':['mean', 'sum', 'std'],
    'toilet_stl1': ['mean', 'sum', 'std'], 
    'toilet_stl2':['mean', 'sum', 'std'],
    'toilet_stl3':['mean', 'sum', 'std'],
    'toilet_stl4':['mean', 'sum', 'std'],
    'toilet_ssro':['mean', 'sum', 'std'],
    'toilet_slhf':['mean', 'sum', 'std'],
    'toilet_ssr':['mean', 'sum', 'std'],
    'toilet_sp':['mean', 'sum', 'std'],
    'toilet_sro':['mean', 'sum', 'std'],
    'toilet_sshf':['mean', 'sum', 'std'],
    'toilet_ssrd':['mean', 'sum', 'std'],
    'toilet_strd':['mean', 'sum', 'std'],
    'toilet_e':['mean', 'sum', 'std'],
    'toilet_tp':['mean', 'sum', 'std'],
    'toilet_swvl1':['mean', 'sum', 'std'],
    'toilet_swvl2':['mean', 'sum', 'std'],
    'toilet_swvl3':['mean', 'sum', 'std'],
    'toilet_swvl4':['mean', 'sum', 'std'],

})

print(len(merged_data))


lender_agg.columns = ['lender_' + '_'.join(col) for col in lender_agg.columns]
lender_agg.reset_index(inplace=True)
merged_data = merged_data.merge(lender_agg, on='Category_Health_Facility_UUID', how='left')

print(len(merged_data))

tbl_agg = merged_data.groupby('Disease').agg({
    'toilet_10u': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_2d': ['mean', 'max', 'min', 'std'],
    'toilet_2t': ['mean', 'std', 'max', 'min'],
    'toilet_evabs': ['mean', 'sum', 'std'],
    'toilet_evaow': ['mean', 'max', 'min', 'std'],
    'toilet_lshf': ['mean', 'sum', 'std'],
    'toilet_lai_hv': ['mean', 'sum', 'std'],
    'toilet_lai_lv': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'], 
    'toilet_swvl4': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'],
    'toilet_ro': ['mean', 'sum', 'std'],
    'toilet_src':['mean', 'sum', 'std'],
    'toilet_skt': ['mean', 'sum', 'std'],
    'toilet_es':['mean', 'sum', 'std'],
    'toilet_stl1': ['mean', 'sum', 'std'], 
    'toilet_stl2':['mean', 'sum', 'std'],
    'toilet_stl3':['mean', 'sum', 'std'],
    'toilet_stl4':['mean', 'sum', 'std'],
    'toilet_ssro':['mean', 'sum', 'std'],
    'toilet_slhf':['mean', 'sum', 'std'],
    'toilet_ssr':['mean', 'sum', 'std'],
    'toilet_sp':['mean', 'sum', 'std'],
    'toilet_sro':['mean', 'sum', 'std'],
    'toilet_sshf':['mean', 'sum', 'std'],
    'toilet_ssrd':['mean', 'sum', 'std'],
    'toilet_strd':['mean', 'sum', 'std'],
    'toilet_e':['mean', 'sum', 'std'],
    'toilet_tp':['mean', 'sum', 'std'],
    'toilet_swvl1':['mean', 'sum', 'std'],
    'toilet_swvl2':['mean', 'sum', 'std'],
    'toilet_swvl3':['mean', 'sum', 'std'],
    'toilet_swvl4':['mean', 'sum', 'std'],
})


tbl_agg.columns = ['tbl_' + '_'.join(col) for col in tbl_agg.columns]
tbl_agg.reset_index(inplace=True)
merged_data = merged_data.merge(tbl_agg, on='Disease', how='left')




lender_agg_water = merged_data.groupby('Category_Health_Facility_UUID').agg({
    'water_10u':['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_2d': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_2t': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evabs': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evaow': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evatc': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evavt': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_albedo': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lshf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lai_hv': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lai_lv': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_pev' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_src': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_skt': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_es': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl1': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl2': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl3': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl4': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_slhf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssr': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_str': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sp': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sshf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssrd': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_strd': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_e' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_tp': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl1' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl2': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl3': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl4' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    })


lender_agg_water.columns = ['lender_water_' + '_'.join(col) for col in lender_agg_water.columns]
lender_agg_water.reset_index(inplace=True)
merged_data = merged_data.merge(lender_agg_water, on='Category_Health_Facility_UUID', how='left')



tbl_agg_water = merged_data.groupby('Disease').agg({
    'water_10u':['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_2d': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_2t': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evabs': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evaow': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evatc': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evavt': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_albedo': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lshf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lai_hv': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lai_lv': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_pev' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_src': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_skt': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_es': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl1': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl2': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl3': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl4': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_slhf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssr': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_str': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sp': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sshf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssrd': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_strd': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_e' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_tp': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl1' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl2': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl3': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl4' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    })


tbl_agg_water.columns = ['tbl__water' + '_'.join(col) for col in tbl_agg_water.columns]
tbl_agg_water.reset_index(inplace=True)
merged_data = merged_data.merge(tbl_agg_water, on='Disease', how='left')



location_agg_toilet = merged_data.groupby('Location').agg({
    'toilet_10u': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_2d': ['mean', 'max', 'min', 'std'],
    'toilet_2t': ['mean', 'std', 'max', 'min'],
    'toilet_evabs': ['mean', 'sum', 'std'],
    'toilet_evaow': ['mean', 'max', 'min', 'std'],
    'toilet_lshf': ['mean', 'sum', 'std'],
    'toilet_lai_hv': ['mean', 'sum', 'std'],
    'toilet_lai_lv': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'], 
    'toilet_swvl4': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'],
    'toilet_ro': ['mean', 'sum', 'std'],
    'toilet_src':['mean', 'sum', 'std'],
    'toilet_skt': ['mean', 'sum', 'std'],
    'toilet_es':['mean', 'sum', 'std'],
    'toilet_stl1': ['mean', 'sum', 'std'], 
    'toilet_stl2':['mean', 'sum', 'std'],
    'toilet_stl3':['mean', 'sum', 'std'],
    'toilet_stl4':['mean', 'sum', 'std'],
    'toilet_ssro':['mean', 'sum', 'std'],
    'toilet_slhf':['mean', 'sum', 'std'],
    'toilet_ssr':['mean', 'sum', 'std'],
    'toilet_sp':['mean', 'sum', 'std'],
    'toilet_sro':['mean', 'sum', 'std'],
    'toilet_sshf':['mean', 'sum', 'std'],
    'toilet_ssrd':['mean', 'sum', 'std'],
    'toilet_strd':['mean', 'sum', 'std'],
    'toilet_e':['mean', 'sum', 'std'],
    'toilet_tp':['mean', 'sum', 'std'],
    'toilet_swvl1':['mean', 'sum', 'std'],
    'toilet_swvl2':['mean', 'sum', 'std'],
    'toilet_swvl3':['mean', 'sum', 'std'],
    'toilet_swvl4':['mean', 'sum', 'std'],

})


location_agg_toilet.columns = ['tbl_location_toilet' + '_'.join(col) for col in location_agg_toilet.columns]
location_agg_toilet.reset_index(inplace=True)
merged_data = merged_data.merge(location_agg_toilet, on='Location', how='left')

print(len(merged_data))


lender_agg_location_water = merged_data.groupby('Location').agg({
    'water_10u':['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_2d': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_2t': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evabs': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evaow': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evatc': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_evavt': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_albedo': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lshf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lai_hv': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_lai_lv': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_pev' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_src': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_skt': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_es': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl1': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl2': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl3': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_stl4': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_slhf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssr': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_str': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sp': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sro': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_sshf': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_ssrd': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_strd': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_e' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_tp': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl1' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl2': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl3': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'water_swvl4' : ['mean', 'sum', 'max', 'std', 'median', 'min'],
 
})


location_agg_water = merged_data.groupby('Location').agg({
    'toilet_10u': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_10v': ['mean', 'sum', 'max', 'std', 'median', 'min'],
    'toilet_2d': ['mean', 'max', 'min', 'std'],
    'toilet_2t': ['mean', 'std', 'max', 'min'],
    'toilet_evabs': ['mean', 'sum', 'std'],
    'toilet_evaow': ['mean', 'max', 'min', 'std'],
    'toilet_lshf': ['mean', 'sum', 'std'],
    'toilet_lai_hv': ['mean', 'sum', 'std'],
    'toilet_lai_lv': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'], 
    'toilet_swvl4': ['mean', 'sum', 'std'],
    'toilet_pev': ['mean', 'sum', 'std'],
    'toilet_ro': ['mean', 'sum', 'std'],
    'toilet_src':['mean', 'sum', 'std'],
    'toilet_skt': ['mean', 'sum', 'std'],
    'toilet_es':['mean', 'sum', 'std'],
    'toilet_stl1': ['mean', 'sum', 'std'], 
    'toilet_stl2':['mean', 'sum', 'std'],
    'toilet_stl3':['mean', 'sum', 'std'],
    'toilet_stl4':['mean', 'sum', 'std'],
    'toilet_ssro':['mean', 'sum', 'std'],
    'toilet_slhf':['mean', 'sum', 'std'],
    'toilet_ssr':['mean', 'sum', 'std'],
    'toilet_sp':['mean', 'sum', 'std'],
    'toilet_sro':['mean', 'sum', 'std'],
    'toilet_sshf':['mean', 'sum', 'std'],
    'toilet_ssrd':['mean', 'sum', 'std'],
    'toilet_strd':['mean', 'sum', 'std'],
    'toilet_e':['mean', 'sum', 'std'],
    'toilet_tp':['mean', 'sum', 'std'],
    'toilet_swvl1':['mean', 'sum', 'std'],
    'toilet_swvl2':['mean', 'sum', 'std'],
    'toilet_swvl3':['mean', 'sum', 'std'],
    'toilet_swvl4':['mean', 'sum', 'std']})



location_agg_water.columns = ['tbl_location_water' + '_'.join(col) for col in location_agg_water.columns]
location_agg_water.reset_index(inplace=True)
merged_data = merged_data.merge(location_agg_water, on='Location', how='left')


print(len(merged_data))


cluster_features = merged_data[water_cols].fillna(0)
kmeans = KMeans(n_clusters=5, random_state=42).fit(cluster_features)
merged_data['customer_behavior_cluster_toilet'] = kmeans.labels_



cluster_features_toilet = merged_data[toilet_cols].fillna(0)
kmeans = KMeans(n_clusters=5, random_state=42).fit(cluster_features_toilet)
merged_data['customer_behavior_cluster_water'] = kmeans.labels_


cluster_features_cat_cols = merged_data[cat_cols].fillna(0)
kmeans = KMeans(n_clusters=5, random_state=42).fit(cluster_features_toilet)
merged_data['customer_behavior_cluster_cats_'] = kmeans.labels_


#features_for_modelling = [col for col in train_df.columns if col not in date_cols + ['ID']]

train_df = merged_data[merged_data['Year'] < 2023]
test_df = merged_data[merged_data['Year'] == 2023]


print(test_df['ID'].head(10))
print(test['ID'].head(10))


matched_rows = test[test['ID'].isin(merged_data['ID'])]
print(len(matched_rows))
print(len(test))

# Specify the target column
target_column = 'Total'

# Feature and target split
y = train_df[target_column]
X = train_df.drop(columns=['ID', 'Location','Total'])  # Exclude unnecessary columns
test_df = test_df.drop(columns=['ID', 'Location'])  # Exclude unnecessary columns




X.replace([np.inf, -np.inf], 0, inplace=True)
X=X.fillna(0)

test_df.replace([np.inf, -np.inf], 0, inplace=True)
test_df=test_df.fillna(0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForest Regressor
rf_regressor = RandomForestRegressor(random_state=42)

# Perform Randomized Search
rf_random = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_grid,
                               n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
rf_random.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_random.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)**0.5  # RMSE
print(f'Root Mean Squared Error: {mse}')

# Get the best parameters
print(f'Best parameters found: {rf_random.best_params_}')

scores = cross_val_score(rf_random.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
rmse_scores = (-scores)**0.5
print(f'Cross-validated RMSE scores: {rmse_scores}')
print(f'Mean Cross-validated RMSE: {rmse_scores.mean()}')


best_params = rf_random.best_params_

rf_best = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    max_features=best_params['max_features'],
    max_depth=best_params['max_depth'],
    bootstrap=best_params['bootstrap'],
    random_state=42  # Keep the same random state for reproducibility
)

# Fit the model on the entire dataset
rf_best.fit(X_train, y_train)

# Output to check if the model is fitted
print("Model trained on the entire dataset with best parameters.")

y_pred = rf_best.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

print(len(test_df.columns))
print(len(X.columns))


predictions = rf_best.predict(test_df)

sub = test_df[['ID']].copy()
sub['Predicted_Total'] = predictions

sub.to_csv('subs_m9.csv', index=False)
# Define the objective function for Optuna

