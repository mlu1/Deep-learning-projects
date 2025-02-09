import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import optuna  # For hyperparameter optimization
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv('Train.csv')
test=pd.read_csv('Test.csv')
check=pd.read_csv('economy.csv')
print(train.shape)
print(test.shape)
print(check.shape)



inflation_kenya_2021 = check.loc[(check['Country'] == 'Kenya') & (check['Indicator'] == 'Inflation, consumer prices (annual %)'), 'YR2021'].iloc[0]
inflation_kenya_2022 = check.loc[(check['Country'] == 'Kenya') & (check['Indicator'] == 'Inflation, consumer prices (annual %)'), 'YR2022'].iloc[0]
inflation_kenya_2023 = check.loc[(check['Country'] == 'Kenya') & (check['Indicator'] == 'Inflation, consumer prices (annual %)'), 'YR2023'].iloc[0]

inflation_Ghana_2021 = check.loc[(check['Country'] == 'Ghana') & (check['Indicator'] == 'Inflation, consumer prices (annual %)'), 'YR2021'].iloc[0]
inflation_Ghana_2022 = check.loc[(check['Country'] == 'Ghana') & (check['Indicator'] == 'Inflation, consumer prices (annual %)'), 'YR2022'].iloc[0]
inflation_Ghana_2023 = check.loc[(check['Country'] == 'Ghana') & (check['Indicator'] == 'Inflation, consumer prices (annual %)'), 'YR2023'].iloc[0]

train['disbursement_number']=None
train['due_number']=None
for index,row in train.iterrows():
    value_1=row['disbursement_date'].split('-')
    value_2=row['due_date'].split('-')
    train.at[index,'disbursement_number']=int(value_1[0]+value_1[1]+value_1[2])
    train.at[index,'due_number']=int(value_2[0]+value_2[1]+value_2[2])


train['occurrence_number'] = train.groupby('customer_id')['customer_id'].transform('count')




# Step 1: Find the maximum disbursement_number for each customer_id
max_disbursement = train.groupby('customer_id')['disbursement_number'].transform('max')

# Step 2: Filter rows where due_number >= the max disbursement_number for the group
due_gte_max_disbursement = train[train['due_number'] >= max_disbursement]

# Step 3: Get rows with the maximum disbursement_number for each customer_id
max_disbursement_rows = train.loc[train.groupby('customer_id')['disbursement_number'].idxmax()]

# Step 4: Combine the two sets of rows
new_train = pd.concat([max_disbursement_rows, due_gte_max_disbursement]).drop_duplicates()

# Sort the result for better readability (optional)
new_train = new_train.sort_values(by=['customer_id', 'disbursement_number'], ascending=[True, False])

new_train.drop(columns=['disbursement_number'], inplace=True)
new_train.drop(columns=['due_number'], inplace=True)

print(new_train.shape)



test['disbursement_number']=None
test['due_number']=None
for index,row in test.iterrows():
    value_1=row['disbursement_date'].split('-')
    value_2=row['due_date'].split('-')
    test.at[index,'disbursement_number']=int(value_1[0]+value_1[1]+value_1[2])
    test.at[index,'due_number']=int(value_2[0]+value_2[1]+value_2[2])


test['occurrence_number'] = test.groupby('customer_id')['customer_id'].transform('count')




# Step 1: Find the maximum disbursement_number for each customer_id
max_disbursement = test.groupby('customer_id')['disbursement_number'].transform('max')

# Step 2: Filter rows where due_number >= the max disbursement_number for the group
due_gte_max_disbursement = test[test['due_number'] >= max_disbursement]

# Step 3: Get rows with the maximum disbursement_number for each customer_id
max_disbursement_rows = test.loc[test.groupby('customer_id')['disbursement_number'].idxmax()]

# Step 4: Combine the two sets of rows
new_test = pd.concat([max_disbursement_rows, due_gte_max_disbursement]).drop_duplicates()

# Sort the result for better readability (optional)
new_test = new_test.sort_values(by=['customer_id', 'disbursement_number'], ascending=[True, False])

new_test.drop(columns=['disbursement_number'], inplace=True)
new_test.drop(columns=['due_number'], inplace=True)


print(new_test.shape)



data = pd.concat([new_train, new_test]).reset_index(drop=True)


data['date']=None
for index,row in data.iterrows():
    check=row['disbursement_date'].split("-")
    data.at[index,'date']=int(check[0]+check[1]+check[2])
data['disbursement_year']=None
data['disbursement_month']=None
for index,row in data.iterrows():
    check=row['disbursement_date'].split('-')
    data.at[index,'disbursement_year']=int(check[0])
    data.at[index,'disbursement_month']=int(check[1])
data['due_year']=None
data['due_month']=None
for index,row in data.iterrows():
    check=row['due_date'].split('-')
    data.at[index,'due_year']=int(check[0])
    data.at[index,'due_month']=int(check[1])
data['due_year'] = data['due_year'].astype(int)
data['disbursement_year'] = data['disbursement_year'].astype(int)
data['temp_order'] = data.groupby('customer_id')['date'].rank(method='dense').astype(int)
data = data.sort_values(['customer_id', 'temp_order']).drop(columns=['temp_order'])

data['order'] = data.groupby('customer_id')['date'].rank(method='dense', ascending=True).astype(int)

unique_customer_ids = data['customer_id'].unique()
customer_id_mapping = {customer_id: idx + 1 for idx, customer_id in enumerate(sorted(unique_customer_ids))}

# Step 2: Apply the mapping to the customer_id column
data['customer_id'] = data['customer_id'].map(customer_id_mapping)

unique_customer_ids = data['tbl_loan_id'].unique()
customer_id_mapping = {customer_id: idx + 1 for idx, customer_id in enumerate(sorted(unique_customer_ids))}

# Step 2: Apply the mapping to the customer_id column
data['tbl_loan_id'] = data['tbl_loan_id'].map(customer_id_mapping)


for index,row in data.iterrows():
    check=row['loan_type'].split('_')
    data.at[index,'loan_type']=int(check[1])

unique_lender_ids = data['lender_id'].unique()
lender_id_mapping = {lender_id: idx + 1 for idx, lender_id in enumerate(sorted(unique_lender_ids))}

# Step 2: Apply the mapping to the lender_id column
data['lender_id'] = data['lender_id'].map(lender_id_mapping)


data['year']=None
for index,row in data.iterrows():
    check=row['disbursement_date'].split('-')
    data.at[index,'year']=int(check[0])


data = data.drop(columns=['disbursement_month','lender_id','due_month','country_id','date','New_versus_Repeat'])
data['loan_type'] = data['loan_type'].astype(int)
data['year'] = data['year'].astype(int)
from sklearn.preprocessing import LabelEncoder
# Combine datasets for consistent feature engineering
date_cols = ['disbursement_date', 'due_date']

# Create financial ratios and transformations
data['repayment_ratio'] = data['Total_Amount_to_Repay'] / data['Total_Amount']
data['log_Total_Amount'] = np.log1p(data['Total_Amount'])
#data = data.drop(columns=['Lender_portion_to_be_repaid','Total_Amount_to_Repay'])
# Split back into train and test
data = data.drop(columns=['disbursement_year', 'due_year'])
train_df = data[data['ID'].isin(train['ID'].unique())]
test_df = data[data['ID'].isin(test['ID'].unique())]
# Define features for modeling
features_for_modelling = [col for col in train_df.columns if col not in date_cols + ['ID', 'target']]

print(f"The shape of train_df is: {train_df.shape}")
print(f"The shape of test_df is: {test_df.shape}")
print(f"The shape of train is: {train.shape}")
print(f"The shape of test is: {test.shape}")
print(f"The features for modelling are:\n{features_for_modelling}")



train_df = train_df.set_index('ID').reindex(train['ID']).reset_index()


X_train, X_valid, y_train, y_valid = train_test_split(
    train_df[features_for_modelling],
    train['target'],
    stratify=train['target'],
    shuffle=True,
    random_state=42
)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


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
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
    
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
    final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
elif best_params["booster"] == "lightgbm":
    final_model = LGBMClassifier(**best_params, random_state=42)
elif best_params["booster"] == "catboost":
    del best_params["booster"]
    final_model = CatBoostClassifier(**best_params, random_state=42, verbose=0)

# Train and evaluate final model
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_valid)
y_pred_proba = final_model.predict_proba(X_valid)[:, 1]

# Print model performance metrics
f1 = f1_score(y_valid, y_pred)
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_valid,
    y_pred,
    display_labels=final_model.classes_,
    cmap=plt.cm.Blues
)

test_predictions = final_model.predict(test_df[features_for_modelling])
test_predictions_proba = final_model.predict_proba(test_df[features_for_modelling])[:, 1]

# Create submission dataframe
test_df['target'] = test_predictions
test_df['credit_score'] = test_predictions_proba  # Add probability column

# Select required columns
sub = test_df[['ID', 'target', 'credit_score']]
sub.shape



s=0
for index,row in sub.iterrows():
    if row['target']==1:
        s+=1
print(s)


data = pd.concat([new_train, new_test]).reset_index(drop=True)
data['date']=None
for index,row in data.iterrows():
    check=row['disbursement_date'].split("-")
    data.at[index,'date']=int(check[0]+check[1]+check[2])
data['temp_order'] = data.groupby('customer_id')['date'].rank(method='dense').astype(int)
data = data.sort_values(['customer_id', 'temp_order']).drop(columns=['temp_order'])
data['order'] = data.groupby('customer_id')['date'].rank(method='dense', ascending=True).astype(int)
unique_customer_ids = data['customer_id'].unique()
customer_id_mapping = {customer_id: idx + 1 for idx, customer_id in enumerate(sorted(unique_customer_ids))}

# Step 2: Apply the mapping to the customer_id column
data['customer_id'] = data['customer_id'].map(customer_id_mapping)
unique_customer_ids = data['tbl_loan_id'].unique()
customer_id_mapping = {customer_id: idx + 1 for idx, customer_id in enumerate(sorted(unique_customer_ids))}

# Step 2: Apply the mapping to the customer_id column
data['tbl_loan_id'] = data['tbl_loan_id'].map(customer_id_mapping)
for index,row in data.iterrows():
    check=row['loan_type'].split('_')
    data.at[index,'loan_type']=int(check[1])
unique_lender_ids = data['lender_id'].unique()
lender_id_mapping = {lender_id: idx + 1 for idx, lender_id in enumerate(sorted(unique_lender_ids))}

# Step 2: Apply the mapping to the lender_id column
data['lender_id'] = data['lender_id'].map(lender_id_mapping)
data['disbursement_year']=None
data['disbursement_month']=None
for index,row in data.iterrows():
    check=row['disbursement_date'].split('-')
    data.at[index,'disbursement_year']=int(check[0])
    data.at[index,'disbursement_month']=int(check[1])
data['due_year']=None
data['due_month']=None
for index,row in data.iterrows():
    check=row['due_date'].split('-')
    data.at[index,'due_year']=int(check[0])
    data.at[index,'due_month']=int(check[1])

columns_to_update1 = [
    'Amount_Funded_By_Lender',
    'Total_Amount',
]

columns_to_update2 = [
    'Lender_portion_to_be_repaid',
    'Total_Amount_to_Repay',
]

# Step 1: Count occurrences of each 'tbl_loan_id'
loan_counts = data['tbl_loan_id'].value_counts()
#data=data[data['disbursement_year']!=2024]
data['amount_jdid']=data['Total_Amount']

# Step 2: Map the counts to the DataFrame and assign 'tbl_loan_type'
data['tbl_loan_type'] = data['tbl_loan_id'].map(lambda x: 1 if loan_counts[x] == 1 else 2)
#data['disbursement_year'] = data['disbursement_year'].astype(int)


data.loc[(data['country_id'] == 'Ghana') & (data['due_year'] == 2021) , columns_to_update2] *=(1+inflation_Ghana_2021/100)
data.loc[(data['country_id'] == 'Ghana') & (data['due_year'] == 2022) , columns_to_update2] *=(1+inflation_Ghana_2022/100)
data.loc[(data['country_id'] == 'Ghana') & (data['due_year'] == 2023) , columns_to_update2] *=(1+inflation_Ghana_2023/100)
data.loc[(data['country_id'] == 'Ghana') & (data['due_year'] == 2024)  , columns_to_update2] *=(1+(inflation_Ghana_2023-9)/100)



data.loc[(data['country_id'] == 'Kenya') & (data['due_year'] == 2021)  , columns_to_update2] *=(1+inflation_kenya_2021/100)
data.loc[(data['country_id'] == 'Kenya') & (data['due_year'] == 2022) , columns_to_update2] *=(1+inflation_kenya_2022/100)
data.loc[(data['country_id'] == 'Kenya') & (data['due_year'] == 2023)  , columns_to_update2] *=(1+inflation_kenya_2023/100)
data.loc[(data['country_id'] == 'Kenya') & (data['due_year'] == 2024)  , columns_to_update2] *=(1+(inflation_kenya_2023-2)/100)


data = data.drop(columns=['disbursement_date','due_date','lender_id','New_versus_Repeat','date','loan_type','duration','due_year','disbursement_year'])
from sklearn.preprocessing import LabelEncoder
# Combine datasets for consistent feature engineering
date_cols = ['disbursement_date', 'due_date']

# Create financial ratios and transformations
data['repayment_ratio'] = data['Total_Amount_to_Repay'] / data['Total_Amount']
data['log_Total_Amount'] = np.log1p(data['Total_Amount'])
#data['amount_ratio'] = data['Total_Amount_Sum'] / data['Total_Amount']
#data['log_Total_Amount'] = np.log1p(data['Amount_Funded_By_Lender'])
#data['repayment_ratio'] = data['Lender_portion_to_be_repaid'] / data['Amount_Funded_By_Lender']
#data = data.drop(columns=['Lender_portion_to_be_repaid','Total_Amount_to_Repay'])
# Split back into train and test
train_df = data[data['ID'].isin(train['ID'].unique())]
test_df = data[data['ID'].isin(test['ID'].unique())]
test_df=test_df[test_df['country_id']=='Ghana']
test_df = test_df.drop(columns=['Total_Amount', 'Total_Amount_to_Repay', 'Amount_Funded_By_Lender', 'Lender_portion_to_be_repaid','disbursement_month','due_month','tbl_loan_type','tbl_loan_id','country_id','amount_jdid'])
train_df = train_df.drop(columns=['Total_Amount','Total_Amount_to_Repay', 'Amount_Funded_By_Lender','Lender_portion_to_be_repaid','disbursement_month','due_month','tbl_loan_type','tbl_loan_id','country_id','amount_jdid'])
# Define features for modeling
features_for_modelling = [col for col in train_df.columns if col not in date_cols + ['ID', 'target']]

print(f"The shape of train_df is: {train_df.shape}")
print(f"The shape of test_df is: {test_df.shape}")

print(f"The shape of train is: {train.shape}")
print(f"The shape of test is: {test.shape}")
print(f"The features for modelling are:\n{features_for_modelling}")


train_df = train_df.set_index('ID').reindex(train['ID']).reset_index()
# Create stratified train-validation split
X_train, X_valid, y_train, y_valid = train_test_split(
    train_df[features_for_modelling], 
    train['target'], 
    stratify=train['target'], 
    shuffle=True, 
    random_state=42
)
print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)


import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Optuna objective function for hyperparameter tuning
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
        model = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")

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

    # Train model
    model.fit(X_train, y_train)

    # Validate model
    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)

    return f1

# Optimize hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Get best hyperparameters
best_params = study.best_params
booster = best_params.pop("booster")  # Remove 'booster' before passing to models

# Train final model with best parameters
if booster == "xgboost":
    final_model1 = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="logloss", random_state=42)
elif booster == "lightgbm":
    final_model1 = LGBMClassifier(**best_params, random_state=42)
elif booster == "catboost":
    final_model1 = CatBoostClassifier(**best_params, random_state=42, verbose=0)

# Train and evaluate final model
final_model1.fit(X_train, y_train)
y_pred = final_model1.predict(X_valid)
y_pred_proba = final_model1.predict_proba(X_valid)[:, 1]

# Print model performance metrics
f1 = f1_score(y_valid, y_pred)
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_valid, y_pred))

# Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_valid,
    y_pred,
    display_labels=final_model.classes_,
    cmap=plt.cm.Blues
)
plt.title("Confusion Matrix")
plt.show()


feature_importances = final_model1.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features_for_modelling,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.title('Top 20 Feature Importances')
plt.show()

test_predictions = final_model1.predict(test_df[features_for_modelling])
test_predictions_proba = final_model1.predict_proba(test_df[features_for_modelling])[:, 1]

# Create submission dataframe
test_df['target'] = test_predictions
test_df['credit_score'] = test_predictions_proba  # Add probability column

# Select required columns
sub2 = test_df[['ID', 'target', 'credit_score']]
sub2.shape


s=0
for index,row in test_df.iterrows():
    if row['target']==1:
        s+=1
print(s)



for index, row in sub.iterrows():
    # Check if the ID in df_11 exists in submission2
    matching_row = sub2[sub2['ID'] == row['ID']]

    # If a match is found, update the 'target' in df_11
    if not matching_row.empty:
        sub.at[index, 'target'] = matching_row['target'].values[0]
        sub.at[index, 'credit_score'] = matching_row['credit_score'].values[0]


df_11 = pd.DataFrame({'ID': pd.Series(dtype='str'),'target': pd.Series(dtype='int'),'credit_score':pd.Series(dtype='float')})
def add_row_1(ID,target,credit_score):
    global df_11
    new_row = pd.DataFrame({'ID':[ID],'target': [target],'credit_score':[credit_score]})
    df_11 = pd.concat([df_11, new_row], ignore_index=True)
df=pd.read_csv('Test.csv')
for index,row in df.iterrows():
    add_row_1(row['ID'],0,0.0)
# Iterate through df_11 and update the 'target' value where IDs match
for index, row in df_11.iterrows():
    # Check if the ID in df_11 exists in submission2
    matching_row = sub[sub['ID'] == row['ID']]
    
    # If a match is found, update the 'target' in df_11
    if not matching_row.empty:
        df_11.at[index, 'target'] = matching_row['target'].values[0]
        df_11.at[index, 'credit_score'] = matching_row['credit_score'].values[0]
df_11.shape
(18594, 3)
df_11['target'] = df_11['target'].astype(int)
s=0
for index,row in df_11.iterrows():
    if row['target']==1:
        s+=1
print(s)



def calculate_credit_score_single_row(model, row, features_for_modelling):
    """
    Calculates the credit score for a single row based on the given model's prediction.

    Parameters:
    - model (sklearn model): The trained model (final_model or final_model1).
    - row (Series): The single row of data (one observation).
    - features_for_modelling (list): The features used for prediction.

    Returns:
    - credit_score (float): The probability (credit score) for the given row.
    """

    # Prepare the row as a DataFrame (model expects a 2D array-like structure)
    row_df = pd.DataFrame([row[features_for_modelling]])

    # Predict the probability of default (class 1)
    credit_score = model.predict_proba(row_df)[:, 1][0]  # Probability for the positive class (default)

    return credit_score

# Example usage:
# Assuming 'row' is a single row from test_df and 'final_model' is the model you want to use
row = test_df.iloc[0]  # Just as an example, picking the first row
credit_score = calculate_credit_score_single_row(final_model1, row, features_for_modelling)

print(f"Credit score for this row: {credit_score}")
zero_count = (df_11['credit_score'] == 0.0).sum()
between_count = ((df_11['credit_score'] > 0.0) & (df_11['credit_score'] < 0.9)).sum()
above_count = (df_11['credit_score'] >= 0.9).sum()

print(f"Count of credit_score == 0.0: {zero_count}")
print(f"Count of 0.0 < credit_score < 0.9: {between_count}")
print(f"Count of credit_score >= 0.9: {above_count}")
import numpy as np

# Define conditions
conditions = [
    (df_11['credit_score'] == 0.0),
    (df_11['credit_score'] > 0.0) & (df_11['credit_score'] < 0.9),
    (df_11['credit_score'] >= 0.9)
]

# Define category labels
categories = ['Low Risk', 'Moderate Risk', 'High Risk']

# Assign categories
df_11['risk_category'] = np.select(conditions, categories, default='Unknown')

# Display value counts for the new column
print(df_11['risk_category'].value_counts())
df_11 = df_11.drop(columns=['risk_category', 'credit_score'])



