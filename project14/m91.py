import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier,VotingClassifier 
import warnings
from lightgbm import LGBMClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA, FastICA, TruncatedSVD,FactorAnalysis
pca = PCA(random_state=42,n_components=1)
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import optuna
# Load the data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
subs = pd.read_csv('Test.csv')
economy = pd.read_csv("economy.csv")
cpi = pd.read_csv("cpi.csv")
inflation = pd.read_csv("inflation.csv")
forex = pd.read_csv("forex.csv")
nominal = pd.read_csv("nominal.csv")

sum_cols = ['Amount_Funded_By_Lender','Lender_portion_to_be_repaid','Total_Amount_to_Repay','Total_Amount']

# Melt the year columns into a single column for your dataset
df = economy.melt(
    id_vars=["Country", "Indicator"],
    var_name="Year",
    value_name="Value"
)

df["Year"] = df["Year"].str.extract('(\d+)').astype(int)
cpi["date"] = pd.to_datetime(cpi['observation_date'])
cpi['Year'] = cpi["date"].dt.year

inflation["date"] = pd.to_datetime(inflation['observation_date'])
inflation['Year'] = inflation["date"].dt.year

nominal["date"] = pd.to_datetime(nominal['observation_date'])
nominal['Year'] = nominal["date"].dt.year
nominal = nominal.drop(['observation_date','date'],axis =1)


forex["date"] = pd.to_datetime(forex['observation_date'])
forex['Year'] = forex["date"].dt.year
forex = forex.drop(['observation_date','date'],axis =1)


economy_df = df.pivot(
    index=['Country', 'Year'],   # These columns will form the index
    columns='Indicator',        # These unique values become column names
    values='Value'              # The values for each indicator
)

economy_df = economy_df.drop(['Average precipitation in depth (mm per year)','Fossil fuel energy consumption (% of total)'],axis =1)
#print(economy_df.head(10)) 
 

uid  =['lender_id','customer_id','tbl_loan_id'] 
cat_1_features =train.filter(uid)
all_cat = cat_1_features
train_pca = pca.fit_transform(all_cat)
train['PCA_CAT'] = train_pca[:,0]

cat_2_features =test.filter(uid)
all_cat_test = cat_2_features
test_pca = pca.fit_transform(all_cat_test)
test['PCA_CAT'] = test_pca[:,0]
###################################################################

coords = np.vstack((train[['Total_Amount','Total_Amount_to_Repay']].values,
                    train[['Total_Amount','Total_Amount_to_Repay']].values))

from sklearn.decomposition import PCA
pca = PCA(random_state=50).fit(coords)
train['pickup_pca0'] = pca.transform(train[['Total_Amount','Total_Amount_to_Repay']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['Total_Amount','Total_Amount_to_Repay']])[:, 1]

test['pickup_pca0'] = pca.transform(test[['Total_Amount','Total_Amount_to_Repay']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['Total_Amount','Total_Amount_to_Repay']])[:, 1]


def age_ranges(data):
    age=[]
    for i in range(0,len(data)):
        x=data['Total_Amount'].iloc[i]
        if x < 10:
            age.append(0)
        elif x >= 10 and x <100:
            age.append(1)
        elif x >= 100 and x <1000:
            age.append(2)
        elif x >=1001 and x < 2000:
            age.append(3)
        elif x >= 2001 and x <3000:
            age.append(4)
        elif x >= 3001:
            age.append(5)
        else:
            age.append(6)

    return age


kmeans = KMeans(n_clusters=2, random_state=42)
train['Cluster'] = kmeans.fit_predict(train[['customer_id', 'lender_id']])
test['Cluster'] = kmeans.predict(test[['customer_id', 'lender_id']])


train['total_repayment_progress'] = (train['Total_Amount_to_Repay']/(train['duration']/12*52+1))*100
test['total_repayment_progress'] = (test['Total_Amount_to_Repay']/(test['duration']/12*52+1))*100

train['emi_paid_progress_perc'] = ((train['Lender_portion_Funded']/(train['duration']/12*52+1))*100)
test['emi_paid_progress_perc'] = ((test['Lender_portion_Funded']/(test['duration']/12*52+1))*100)


train['emi_paid_progress_perc_1'] = ((train['Total_Amount']/(train['duration']/12*52+1))*100)
test['emi_paid_progress_perc_1'] = ((test['Total_Amount']/(test['duration']/12*52+1))*100)

train['emi_paid_progress_perc_2'] = ((train['Amount_Funded_By_Lender']/(train['duration']/12*52+1))*100)
test['emi_paid_progress_perc_2'] = ((test['Amount_Funded_By_Lender']/(test['duration']/12*52+1))*100)

train['emi_paid_progress_perc_3'] = ((train['emi_paid_progress_perc_2']*(train['emi_paid_progress_perc_1'])))
test['emi_paid_progress_perc_3'] = ((test['emi_paid_progress_perc_2']*(test['emi_paid_progress_perc_1'])))


def difference(df):
    pay = []
    for i in range(0,len(df)):
        if (df['Total_Amount_to_Repay'].iloc[i] - df['Total_Amount'].iloc[i]) > 0:
            pay.append(0)
        elif ((df['Total_Amount_to_Repay'].iloc[i] - df['Total_Amount'].iloc[i]) == 0):
            pay.append(1)
        elif ((df['Total_Amount_to_Repay'].iloc[i] == 0.00 and df['Total_Amount'].iloc[i]) == 0.00):
            pay.append(3)

        else:
            pay.append(2)
    return pay



def lender_difference(df,col1,col2):
    pay = []
    for i in range(0,len(df)):
        if (df[col1].iloc[i] - df[col2].iloc[i]) > 0:
            pay.append(0)
        elif ((df[col1].iloc[i] - df[col2].iloc[i]) == 0):
            pay.append(1)
       
        elif ((df[col1].iloc[i] == 0.00 and df[col2].iloc[i]) == 0.00):
            pay.append(3)
        
        else:
            pay.append(2)
    return pay


lenders = ['Amount_Funded_By_Lender','Lender_portion_Funded','Lender_portion_to_be_repaid']
p_targets = []

# Preprocessing
data = pd.concat([train, test]).reset_index(drop=True)
# Convert the datetime columns appropriately
date_cols = ['disbursement_date','due_date']
for col in date_cols:
    data[col] = pd.to_datetime(data[col])
    data[col+'_month'] = data[col].dt.month.astype(np.int8)
    data[col+'_day'] = data[col].dt.day
    data[col+'_year'] = data[col].dt.year    
    data[col+'is_month_start'] = data[col].dt.is_month_start.astype(int)
    data[col+'quarter_of_year'] = data[col].dt.quarter
    data[col+'is_year_end'] = data[col].dt.is_year_end
    data[col+'is_year_start'] = data[col].dt.is_year_start
    data[col+'is_month_end'] = data[col].dt.is_month_end.astype(np.int8)
    data[col+'is_month_start'] = data[col].dt.is_month_start.astype(int)
    data[col+'Day_sin'] = np.sin(2 * np.pi * data[col+'_day'] / 365)
    data[col+'Day_cos'] = np.cos(2 * np.pi * data[col+'_day'] / 365)
    data[col+'Month_sin'] = np.sin(2 * np.pi * data[col+'_month'] / 12)
    data[col+'Month_cos'] = np.cos(2 * np.pi * data[col+'_month'] / 12)
    data[col+'quarter_of_year'] = data[col].dt.quarter
    data[col+'day_of_year'] = data[col].dt.dayofyear 
    

data['issueDate_year_earliesCreditLine_year_minus'] = data[date_cols[0]+'_year'] - \
    data['disbursement_date_year']



data['TODAYS_DATE'] = pd.Timestamp('20241201')
data['DISBURSAL_DATE_DAYS'] =(data['TODAYS_DATE'] - data[date_cols[1]])
data['DUE_DATE_DAYS'] = (data['TODAYS_DATE'] - data[date_cols[0]])
data['ds_days'] = data['DISBURSAL_DATE_DAYS'].dt.days
data['due_days'] = data['DUE_DATE_DAYS'].dt.days

data = data.rename(columns={'due_date_year': 'Year','country_id':'Country'})
economy_df.columns = economy_df.columns.str.replace(r'[^\w\s]', '', regex=True).str.replace(' ', '_')
data1 = pd.merge(data, economy_df, on=["Country", "Year"], how="left")


print(data1.columns)
print(len(data1.columns))


data2 = pd.merge(cpi, data1, on=["Country", "Year"], how="right")

print(data2.columns)
print(len(data2.columns))



data = pd.merge(inflation, data2, on=["Country", "Year"], how="right")


print(data2.columns)
print(len(data2.columns))


cat_cols = data.select_dtypes(include='object').columns

print(data.head(20))
print(cat_cols)

le = LabelEncoder()
for col in [col for col in cat_cols if col not in ['ID']]:
    data[col] = le.fit_transform(data[col])


for j in range(0,len(data)):
    if ((data[lenders[0]].iloc[j]==0) and (data[lenders[1]].iloc[j]==0) and (data[lenders[2]].iloc[j] == 0)):
        p_targets.append(0)
    else:
        p_targets.append(1)

###############################################################

data['placeID_freq'] = data['lender_id'].map(data['lender_id'].value_counts())
data['custmerID_freq'] = data['customer_id'].map(data['customer_id'].value_counts())
data['loanID_freq'] = data['tbl_loan_id'].map(data['tbl_loan_id'].value_counts())
data['loan_typeID_freq'] = data['loan_type'].map(data['loan_type'].value_counts())
data['loan_New_versus_Repeat'] = data['New_versus_Repeat'].map(data['New_versus_Repeat'].value_counts())
data['loan_Month_dis_freq'] = data['disbursement_date_month'].map(data['disbursement_date_month'].value_counts())
data['loan_Month_due_freq'] = data['due_date_month'].map(data['due_date_month'].value_counts())
data['duration_freq'] = data['duration'].map(data['duration'].value_counts())
data['due_date_freq'] = data['due_date_month'].map(data['due_date_month'].value_counts())
data['disbursement_date_freq'] = data['disbursement_date_month'].map(data['disbursement_date_month'].value_counts())


data['p_target'] = p_targets
data['amount_ratio'] = data['Total_Amount']/data['duration'] 
data['amount_ratio_2'] = data['Total_Amount_to_Repay']/data['duration'] 
data['amount_diff'] = data['Total_Amount']- data['Total_Amount_to_Repay']
data['money_stats_1'] = data['Lender_portion_to_be_repaid']/data['Lender_portion_Funded'] 
data['money_stats_2'] = data['Amount_Funded_By_Lender']-data['Lender_portion_to_be_repaid'] 
data['pays'] = difference(data)
data['lenders'] = lender_difference(data,'Amount_Funded_By_Lender','Lender_portion_Funded')
data['avg'] = (data['Amount_Funded_By_Lender']/data['Lender_portion_Funded'])/(data['Total_Amount']+data['Total_Amount_to_Repay']) 
data['avg_duration'] = data['avg']/data['duration']
data['age_size'] = age_ranges(data)
data['interest'] = (abs(data['Total_Amount_to_Repay'] - data['Total_Amount'])/data['Total_Amount'])*100
data['interest_ratio'] = ((data['Total_Amount_to_Repay'] - data['Total_Amount'])/data['Total_Amount'])/data['duration']
data['interest_avg'] = data['interest'] * data['duration']
data['Repayment_to_Total_Ratio'] =  data['Total_Amount_to_Repay']/data['Total_Amount']
data['New_versus_Repeat'] = data['interest'] * data['Repayment_to_Total_Ratio']
data['Lender_Effective_Interest'] = abs(data['Lender_portion_to_be_repaid'] - data['Lender_portion_Funded'])/data['Lender_portion_Funded']



def quarter_of_month(x):
  if x>=1 and x<=7:
    return 1
  elif x>7 and x<=14:
    return 2
  elif x>14 and x<=21:
    return 3
  else:
    return 4

data["Placement - Day of Month_quarter_of_month"] = data["disbursement_date_month"].apply(quarter_of_month)
data["Placement - due_quarter_of_month"] = data["due_date_month"].apply(quarter_of_month)


data['bad_state_1'] = data['Total_Amount'] + (data['Total_Amount_to_Repay']/data['Amount_Funded_By_Lender']) + (data['money_stats_2']/data['Amount_Funded_By_Lender']) + (data['amount_diff']/data['Amount_Funded_By_Lender']) + (data['amount_ratio']/data['Amount_Funded_By_Lender'])
data['bad_state_2'] = data['Total_Amount_to_Repay'] + (data['Amount_Funded_By_Lender']/data['Lender_portion_Funded']) + (data['money_stats_2']/data['Amount_Funded_By_Lender']) + (data['amount_ratio_2']/data['Amount_Funded_By_Lender']) + (data['New_versus_Repeat']/data['Lender_portion_to_be_repaid'])


data['interest/quarter_days'] = data['interest']/data["Placement - Day of Month_quarter_of_month"]
data['interest/quarter_days'] = data['interest']/data["Placement - due_quarter_of_month"]


data['n_consecutives_PAST_days_WE_ou_ferie'] = data.groupby('loan_type')['Total_Amount'].shift(1).fillna(0).astype(int)
data.loc[:, 'n_consecutives_PAST_days_WE_ou_ferie_lag1'] = data.groupby('loan_type')['Amount_Funded_By_Lender'].shift(1).fillna(0).astype(int)



cols = [
        'Amount_Funded_By_Lender',
        'Lender_portion_to_be_repaid',
        'Total_Amount_to_Repay',
        'New_versus_Repeat' 
        ]


for col in cols:
    data[f'{col}_mean'] = data[f'{col}'].apply(lambda x : np.mean(x))

# Handle categorical columns
# Split the data back into train and test
train_df = data[data['ID'].isin(train['ID'].unique())]
test_df = data[data['ID'].isin(test['ID'].unique())]

print(test_df.head(10))
print(train_df.head(10))


# Drop unnecessary columnsfeatures_for_modelling = [col for col in train_df.columns if col not in date_cols + ['MID', 'target','data_diff','country_id']]

features_for_modelling = [col for col in train_df.columns if col not in date_cols + ['DUE_DATE_DAYS','DISBURSAL_DATE_DAYS','TODAYS_DATE','ID','target','date','observation_date','date_y','date_x','observation_date_y','observation_date_x','data_diff','real_ex_rate']]



train_df = train_df[features_for_modelling]
test_df = test_df[features_for_modelling]

feat_with_skewness=['Total_Amount','Total_Amount_to_Repay']

def transform(df):
    if (set(feat_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
        df[feat_with_skewness] = np.cbrt(df[feat_with_skewness])
        return df
    else:
        print("One or more features are not in the dataframe")
        return df


train_df = transform(train_df)
test_df = transform(test_df)


def encode_AG2( uids,main_columns, train_df=train, test_df=test):
    for main_column in main_columns:  
        for col in uids:
            print(col)
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print(col+'_'+main_column+'_ct, ',end='')


def encode_AG(uids ,main_columns, aggregations, train_df=train, test_df=test, ext_src=None,fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                if ext_src is None: 
                    temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                    new_col_name = main_column+'_'+col+'_'+agg_type
                                    
                else:
                    temp_df = ext_src.copy()
                    new_col_name = "ext_data"+ "_"+main_column+'_'+col+'_'+agg_type

                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')


encode_AG2( ['customer_id'],['lender_id','loan_type','tbl_loan_id'], train_df=train_df, test_df=test_df)
encode_AG(['customer_id'] ,  ['Total_Amount','Total_Amount_to_Repay','Amount_Funded_By_Lender'], ['mean','min','max','std'], train_df=train_df, test_df=test_df,fillna=True, usena=False)

train_df['NO2_total_cluster_mean'] = train_df.groupby(['Cluster'])['Total_Amount'].transform('mean')
train_df['NO2_total_month_mean'] = train_df.groupby(['due_date_month'])['Total_Amount'].transform('mean')
train_df['NO2_total_dis_month_mean'] = train_df.groupby(['disbursement_date_month'])['Total_Amount'].transform('mean')
train_df['NO2_total_customer_loan_mean'] = train_df.groupby(['customer_id'])['Total_Amount'].transform('mean')
train_df['NO2_total_lender_loan_mean'] = train_df.groupby(['lender_id'])['Total_Amount'].transform('mean')
train_df['NO2_total_tbl_loan_id_mean'] = train_df.groupby(['tbl_loan_id'])['Total_Amount'].transform('mean')
train_df['NO2_total_loan_type_mean'] = train_df.groupby(['loan_type'])['Total_Amount'].transform('mean')



train_df['lender_total_cluster_mean'] = train_df.groupby(['Cluster'])['Total_Amount_to_Repay'].transform('mean')
train_df['lender_total_month_mean'] = train_df.groupby(['due_date_month'])['Total_Amount_to_Repay'].transform('mean')
train_df['lender_total_dis_month_mean'] = train_df.groupby(['disbursement_date_month'])['Total_Amount_to_Repay'].transform('mean')
train_df['lender_total_customer_loan_mean'] = train_df.groupby(['customer_id'])['Total_Amount_to_Repay'].transform('mean')
train_df['lender_total_lender_loan_mean'] = train_df.groupby(['lender_id'])['Total_Amount_to_Repay'].transform('mean')
train_df['lender_total_tbl_loan_id_mean'] = train_df.groupby(['tbl_loan_id'])['Total_Amount_to_Repay'].transform('mean')
train_df['NO2_total_loan_type_mean'] = train_df.groupby(['loan_type'])['Total_Amount_to_Repay'].transform('mean')


train_df['tbl_total_cluster_mean'] = train_df.groupby(['Cluster'])['Amount_Funded_By_Lender'].transform('mean')
train_df['tbl_total_month_mean'] = train_df.groupby(['due_date_month'])['Amount_Funded_By_Lender'].transform('mean')
train_df['tbl_total_dis_month_mean'] = train_df.groupby(['disbursement_date_month'])['Amount_Funded_By_Lender'].transform('mean')
train_df['tbl_total_customer_loan_mean'] = train_df.groupby(['customer_id'])['Amount_Funded_By_Lender'].transform('mean')
train_df['tbl_total_lender_loan_mean'] = train_df.groupby(['lender_id'])['Amount_Funded_By_Lender'].transform('mean')
train_df['tbl_total_tbl_loan_id_mean'] = train_df.groupby(['tbl_loan_id'])['Amount_Funded_By_Lender'].transform('mean')
train_df['NO2_total_loan_type_mean'] = train_df.groupby(['loan_type'])['Amount_Funded_By_Lender'].transform('mean')


test_df['NO2_total_cluster_mean'] = test_df.groupby(['Cluster'])['Total_Amount'].transform('mean')
test_df['NO2_total_month_mean'] = test_df.groupby(['due_date_month'])['Total_Amount'].transform('mean')
test_df['NO2_total_dis_month_mean'] = test_df.groupby(['disbursement_date_month'])['Total_Amount'].transform('mean')
test_df['NO2_total_customer_loan_mean'] = test_df.groupby(['customer_id'])['Total_Amount'].transform('mean')
test_df['NO2_total_lender_loan_mean'] = test_df.groupby(['lender_id'])['Total_Amount'].transform('mean')
test_df['NO2_total_tbl_loan_id_mean'] = test_df.groupby(['tbl_loan_id'])['Total_Amount'].transform('mean')
test_df['NO2_total_loan_type_mean'] = test_df.groupby(['loan_type'])['Total_Amount'].transform('mean')



test_df['lender_total_cluster_mean'] = test_df.groupby(['Cluster'])['Total_Amount_to_Repay'].transform('mean')
test_df['lender_total_month_mean'] = test_df.groupby(['due_date_month'])['Total_Amount_to_Repay'].transform('mean')
test_df['lender_total_dis_month_mean'] = test_df.groupby(['disbursement_date_month'])['Total_Amount_to_Repay'].transform('mean')
test_df['lender_total_customer_loan_mean'] = test_df.groupby(['customer_id'])['Total_Amount_to_Repay'].transform('mean')
test_df['lender_total_lender_loan_mean'] = test_df.groupby(['lender_id'])['Total_Amount_to_Repay'].transform('mean')
test_df['lender_total_tbl_loan_id_mean'] = test_df.groupby(['tbl_loan_id'])['Total_Amount_to_Repay'].transform('mean')
test_df['NO2_total_loan_type_mean'] = test_df.groupby(['loan_type'])['Total_Amount_to_Repay'].transform('mean')



test_df['tbl_total_cluster_mean'] = test_df.groupby(['Cluster'])['Amount_Funded_By_Lender'].transform('mean')
test_df['tbl_total_month_mean'] = test_df.groupby(['due_date_month'])['Amount_Funded_By_Lender'].transform('mean')
test_df['tbl_total_dis_month_mean'] = test_df.groupby(['disbursement_date_month'])['Amount_Funded_By_Lender'].transform('mean')
test_df['tbl_total_customer_loan_mean'] = test_df.groupby(['customer_id'])['Amount_Funded_By_Lender'].transform('mean')
test_df['tbl_total_lender_loan_mean'] = test_df.groupby(['lender_id'])['Amount_Funded_By_Lender'].transform('mean')
test_df['tbl_total_tbl_loan_id_mean'] = test_df.groupby(['tbl_loan_id'])['Amount_Funded_By_Lender'].transform('mean')
test_df['NO2_total_loan_type_mean'] = test_df.groupby(['loan_type'])['Amount_Funded_By_Lender'].transform('mean')


def Agg(df,cols1,cols2) :
    for col1 in cols1 :
        for col2 in cols2 :
            df[f"{col1}_{col2}_std"] = df.groupby(col1)[col2].transform('mean') 
            df[f"{col1}_{col2}_std"] = df.groupby(col1)[col2].transform('std')
            df[f"{col1}_{col2}_max"] = df.groupby(col1)[col2].transform('max')
            df[f"{col1}_{col2}_min"] = df.groupby(col1)[col2].transform('min')
            df[f"{col1}_{col2}_median"] = df.groupby(col1)[col2].transform('median')
            df[f"{col1}_{col2}_nunique"] = df.groupby(col1)[col2].transform('nunique')
    return df



data["Placement - Day of Month_quarter_of_month"] = data["disbursement_date_month"].apply(quarter_of_month)
data["Placement - due_quarter_of_month"] = data["due_date_month"].apply(quarter_of_month)

cols_1 = ['Total_Amount','Amount_Funded_By_Lender','Total_Amount_to_Repay']
cols_2 = ['loan_type','customer_id','lender_id','tbl_loan_id','duration','disbursement_date_month','due_date_month','Placement - Day of Month_quarter_of_month','Placement - due_quarter_of_month']


for f in cols_1:
    data['{}_cnt'.format(f)] = data.groupby([f])[
        f].transform('count')

for f1 in cols_1:
    for f2 in cols_1:
        if f1 != f2:
            data['{}_{}_cnt'.format(f1, f2)] = data.groupby([f1, f2])[f].transform('count')


train_df=Agg(train_df,cols_2,cols_1)
test_df=Agg(test_df,cols_2,cols_1)


train_df.replace([np.inf, -np.inf], 0, inplace=True)
train_df=train_df.fillna(0)

test_df.replace([np.inf, -np.inf], 0, inplace=True)
test_df=test_df.fillna(0)


X_train, X_valid, y_train, y_valid = train_test_split( 
    train_df, 
    train['target'], 
    stratify=train['target'], 
    shuffle=True, 
    random_state=42
)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_val_scaled = ss.transform(X_valid)

def objective(trial):
    """
    Optuna objective function for hyperparameter optimization.
    Tries different boosting algorithms with various hyperparameters.
    """
    booster = trial.suggest_categorical("booster", ["xgboost", "lightgbm", "catboost"])

    if booster == "xgboost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            "random_state": 42
        }
        model = xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

    elif booster == "lightgbm":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100, step=10),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 1),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 1),
            "random_state": 42
        }
        model = LGBMClassifier(**params)

    elif booster == "catboost":
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500, step=50),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "random_state": 42,
            "verbose": 0
        }
        model = CatBoostClassifier(**params)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)
    return f1

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("Best Hyperparameters:", study.best_params)

# Train final model with best parameters
best_params = study.best_params
if best_params["booster"] == "xgboost":
    del best_params["booster"]
    final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
elif best_params["booster"] == "lightgbm":
    final_model = LGBMClassifier(**best_params, random_state=42)
elif best_params["booster"] == "catboost":
    del best_params["booster"]
    final_model = CatBoostClassifier(**best_params, random_state=42, verbose=0)


print(final_model)
# Train and evaluate final model
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_valid)
y_pred_proba = final_model.predict_proba(X_valid)[:, 1]

# Print model performance metrics
f1 = f1_score(y_valid, y_pred)
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

test_predictions = final_model.predict(test_df)
test_predictions_proba = final_model.predict_proba(test_df)[:, 1]

# Create submission dataframe
submission = pd.DataFrame({
        'ID': subs['ID'],
        'target': test_predictions
        })
submission_file = f"_subs.csv"
submission.to_csv(submission_file, index=False)


