# === Import Libraries ===
# Data manipulation and analysis

# Settings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from scipy.stats import mode

# Advanced ML Models
from imblearn.over_sampling import BorderlineSMOTE
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier


train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
# Display the first few rows of the datasets and their shape

data = pd.concat([train, test]).reset_index(drop=True)

# Convert date columns to datetime
data['disbursement_date'] = pd.to_datetime(data['disbursement_date'], errors='coerce')
data['due_date'] = pd.to_datetime(data['due_date'], errors='coerce')

# update total amount to repay to the the lender portion to be repaid where total amount to be repaid is 0
data.loc[data['Total_Amount_to_Repay'] == 0, ['Total_Amount_to_Repay']] += data['Lender_portion_to_be_repaid']

# Compute mena and median values by customer for the total amount to repay
aggregates = data.groupby('customer_id')['Total_Amount_to_Repay'].agg(['mean', 'median']).reset_index()
aggregates.rename(columns={'mean': 'Mean_Total_Amount', 'median': 'Median_Total_Amount'}, inplace=True)
data=data.merge(aggregates, on='customer_id', how='left')


# Extract temporal features from dates
date_cols = ['disbursement_date', 'due_date']
for col in date_cols:
    data[col] = pd.to_datetime(data[col])
    # Extract month, day, year
    data[col+'_month'] = data[col].dt.month
    data[col+'_day'] = data[col].dt.day
    data[col+'_year'] = data[col].dt.year
    # Calculate loan term and weekday features
    data[f'loan_term_days'] = (data['due_date'] - data['disbursement_date']).dt.days
    data[f'disbursement_weekday'] = data['disbursement_date'].dt.weekday
    data[f'due_weekday'] = data['due_date'].dt.weekday

# Create some financial ratios and transformations
data['repayment_ratio'] = data['Total_Amount_to_Repay'] / data['Total_Amount']
data['amount_due_per_day'] = (data['Total_Amount_to_Repay'] / data['duration'])
data['log_Total_Amount'] = np.log1p(data['Total_Amount'])
data['log_Total_Amount_to_Repay'] = np.log1p(data['Total_Amount_to_Repay']) 
data['log_Amount_Funded_By_Lender'] = np.log1p(data['Amount_Funded_By_Lender'])
data['log_Lender_portion_to_be_repaid'] = np.log1p(data['Lender_portion_to_be_repaid'])
data['amount_to_repay_greater_than_average']=data['Mean_Total_Amount'] - data['Total_Amount_to_Repay'] 

#some outliers were noticed in the total amount and total amount to repay fields. 
# offset this by using the 90th percentile
q=0.9
data['Total_Amount_to_Repay'] = np.where(data['Total_Amount_to_Repay'] >= data['Total_Amount_to_Repay'].quantile(q), data['Total_Amount_to_Repay'].quantile(q),data['Total_Amount_to_Repay'])
data['Total_Amount'] = np.where(data['Total_Amount'] >= data['Total_Amount'].quantile(q), data['Total_Amount'].quantile(q),data['Total_Amount'])

# Handle categorical variables
cat_cols = data.select_dtypes(include='object').columns

# Label encoding for other categorical columns
le = LabelEncoder()
for col in [col for col in cat_cols if col not in ['loan_type', 'ID']]:
    data[col] = le.fit_transform(data[col])

# Split back into train and test
train_df = (data[data['ID'].isin(train['ID'].unique())]).fillna(0)
test_df = (data[data['ID'].isin(test['ID'].unique())]).fillna(0)

# Define features for modeling
features_for_modelling = [col for col in train_df.columns if col not in date_cols + ['ID', 'target', 'country_id','loan_type']]

print(f"The shape of train_df is: {train_df.shape}")
print(f"The shape of test_df is: {test_df.shape}")
print(f"The shape of train is: {train.shape}")
print(f"The shape of test is: {test.shape}")
print(f"The features for modelling are:\n{features_for_modelling}")


cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)



xgb_best_params= {'n_estimators': 250, 'max_depth': 7, 'learning_rate': 0.09952819042992249, 'subsample': 0.9138294343870095, 'colsample_bytree': 0.6808646076666579, 'gamma': 0.01070807358962328, 'min_child_weight': 1}
lgb_best_params=  {'n_estimators': 250, 'max_depth': 8, 'learning_rate': 0.07587945476302646, 'num_leaves': 70, 'feature_fraction': 0.6624074561769746, 'bagging_fraction': 0.662397808134481, 'lambda_l1': 0.05808361216819946, 'lambda_l2': 0.8661761457749352}
cat_best_params= {'iterations': 500, 'depth': 7, 'learning_rate': 0.09702561867586006}



seed=42

X,y=train_df[features_for_modelling], train_df['target']

cv_reports = []
f1_scores = []
predictions=[]
pred_prob=[]
base_estimator = DecisionTreeClassifier()
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y),1):
    
    X_fold_train = X.iloc[train_idx]
    X_fold_valid = X.iloc[valid_idx]
    y_fold_train = y.iloc[train_idx]
    y_fold_valid = y.iloc[valid_idx]
    
    smote =BorderlineSMOTE(sampling_strategy=0.45, random_state=seed)
    
    # Apply SMOTE only on the training data of this fold
    X_fold_train_smote, y_fold_train_smote = smote.fit_resample(X_fold_train, y_fold_train)
    
    # Calculate scale_pos_weight to mitigate class imbalance
    scale_pos_weight = len(y_fold_train_smote[y_fold_train_smote == 1])*0.75 / len(y_fold_train_smote[y_fold_train_smote == 0]) 
    
    model1 = xgb.XGBClassifier(**xgb_best_params,scale_pos_weight=scale_pos_weight,random_state=seed)
    model2 = LGBMClassifier(**lgb_best_params,scale_pos_weight=scale_pos_weight,random_state=seed)
    model3 = CatBoostClassifier(**cat_best_params,scale_pos_weight=scale_pos_weight,random_state=seed)
    

     
    # Train the model on the SMOTE-balanced fold
    model1.fit(X_fold_train_smote, y_fold_train_smote)
    model2.fit(X_fold_train_smote, y_fold_train_smote)
    model3.fit(X_fold_train_smote, y_fold_train_smote)
   

    # predict on validation set
    pred_1 = model1.predict(X_fold_valid)
    pred_2 = model2.predict(X_fold_valid)
    pred_3 = model3.predict(X_fold_valid)

    predictions = mode([pred_1,pred_2,pred_3], axis=0).mode.flatten()

    report = classification_report(y_fold_valid,predictions, output_dict=True)
    cv_reports.append(report)
        
        # Calculate and store the f1 score for this fold
    f1 = f1_score(predictions, y_fold_valid)
    f1_scores.append(f1)


    print(f"Fold {fold} Classification Report:")
    print(classification_report(y_fold_valid, predictions))
    print(f"Fold {fold} F1 Score: {f1:.4f}")
    print("-" * 50)

    # Summary of CV results
    print(f"Mean F1 Score across folds: {np.mean(f1_scores):.4f}")
 


final_predictions = mode(predictions, axis=0).mode.flatten()
final_pred_proba=np.mean(pred_prob,axis=0)
test_df['target'] = final_predictions
test_df[['proba','proba2']]= final_pred_proba


train_test=pd.concat([train_df,test_df])
train_test=(train_test.sort_values(['customer_id','tbl_loan_id','disbursement_date','duration'])).reset_index(drop=True)
# Post Processing functions

def check_loanid_in_train_data(tbl_loan_id):
    '''
    The function checks loans in the test data and update the target to the
    value available in the training data. The intuition behind this is that 
    a loan can only have one decision (0 or 1) and the target in the training data 
    supercedes that of the test data
    '''
    # Check if tbl_loan_id exists in the train DataFrame
    if tbl_loan_id in train_df['tbl_loan_id'].values:
        df = train_df[train_df['tbl_loan_id'] == tbl_loan_id]
        target=df.target.values[0]
    else:
        target=3  # Returns a 3 if there is no match
    return target

    
def type_3_correction (df):
    '''
    The model performed poorly in the type 3 loan type in the Ghana dataset.
    The distribution for that loan type differs to the distribution in the training data
    e.g. The repayment ratio for this loan type is generally much lower than the training dataset
    making the model to predict 0 for all instances (over 3000 instances).
    From the EDAs, it is impossible that all loans in such a class to have just one target group.
    To offset this, an assumption that loans were being disbursed to the customer untill he fails to pay was made.
    This function executes this assumption
    '''
    df1=df[df['loan_type']=='Type_3']
    customer_id=df1.customer_id.unique()
    for customer in customer_id:
        loan_id=df[df['customer_id']==customer].tail(1).tbl_loan_id.values[0]
        df.loc[df['tbl_loan_id'] == loan_id, 'target'] = 1   
    return df


def correct_loan_ids_with_conflicting_target(df):
    '''
    This function ensures that loans only have one target group (0 or 1 not both)
    even if there is more than one lender.
    '''
    # Find loan_ids with different target values
    loan_ids_with_diff_targets = df.groupby('tbl_loan_id')['target'].transform('nunique') > 1

    # Update target values to 1 for those loan_ids
    df.loc[loan_ids_with_diff_targets, 'target'] = 0
    return df    
# Run through the DataFrame and apply the logic explained in the preceeding cell
for i in range(len(train_test)):
    if pd.notna(train_test.loc[i, 'proba']):
        customer_id=train_test.loc[i, 'customer_id']
            
        tbl_loan_id = train_test.loc[i, 'tbl_loan_id']
        target=check_loanid_in_train_data(tbl_loan_id)
            
        if target <2:
            train_test.loc[i, 'target']=target

train_test_f=type_3_correction (train_test)
correct_loan_ids_with_conflicting_target(train_test_f)
test_df=train_test_f[train_test_f['proba'].notna()]
sub =  test_df[['ID', 'target']]
sub.head()

sub.to_csv('final_submission.csv', index=False)
sub.target.sum()


