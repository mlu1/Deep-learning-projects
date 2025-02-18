import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import lightgbm as lgb
from scipy.stats import hmean
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GroupKFold

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
sns.set_theme(rc={'figure.figsize':(3,2)})


data_path = ""

df_train = pd.read_csv(f"{data_path}Train.csv")
df_train.drop_duplicates(ignore_index=True, inplace=True)
grp_cols = ['ID', 'Location','Category_Health_Facility_UUID', 'Disease', 'Month', 'Year', 'Transformed_Latitude', 'Transformed_Longitude']
df_train = df_train.groupby(grp_cols, as_index=False)[['Total']].max()
df_test = pd.read_csv(f"{data_path}Test.csv")
df_train['is_test'] = 0
df_test['is_test'] = 1
df_total = pd.concat([df_train,df_test],ignore_index=True)


df_total.loc[~df_total['Category_Health_Facility_UUID'].isin(list(set(df_train['Category_Health_Facility_UUID']))), 'Category_Health_Facility_UUID'] = 'other'
df_total = pd.get_dummies(data = df_total, columns=['Category_Health_Facility_UUID'], dtype=int)
del df_total['Category_Health_Facility_UUID_other']

df_month_year = df_total[['Year','Month']].drop_duplicates().sort_values(by=['Year','Month'], ignore_index=True)
df_month_year['month_year'] = df_month_year.index
df_total = df_total.merge(df_month_year, on = ['Year','Month'], how='left')

catcols = ['Location','Disease']

df_total[catcols] = df_total[catcols].astype('category')

df_train = df_total[df_total['Total'].notnull()].reset_index(drop=True)
df_test = df_total[df_total['Total'].isnull()].reset_index(drop=True)
remove_from_feats = ['ID','Total','is_test']
features = [i for i in df_test.columns if i not in remove_from_feats]
print(len(features))
print('Features Used:', features)

Features Used: ['Location', 'Disease', 'Month', 'Year', 'Transformed_Latitude', 'Transformed_Longitude',
                'Category_Health_Facility_UUID_56cd4cbb-23db-4dde-a6ae-9fc1ed7c8662',
                'Category_Health_Facility_UUID_a3761841-2a02-4c17-8589-d35aac4edc24',
                'Category_Health_Facility_UUID_a9280aca-c872-46f5-ada7-4a7cc31cf6ec',
                'Category_Health_Facility_UUID_b7f0a600-e19e-4c65-acb3-e28584dae35b',
                'month_year']


# MAE:  6.585647257903238

print(df_train.shape)
target = 'Total'
id_col = 'ID'
print('Features Used:', len(features))

df_test_pred = df_test[[id_col]].copy()
df_oof_pred = []
df_imp = pd.DataFrame()
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
callbacks = [lgb.log_evaluation(period=2000), lgb.early_stopping(stopping_rounds=500)]
for fold, (train_ids, test_ids) in tqdm(enumerate(folds.split(X=df_train, y=df_train['month_year']))):  #pd.cut(df_train[target],10, labels=[i for i in range(10)])
    print('\n●●●●●●●●●●●● Fold :', fold+1, '●●●●●●●●●●●●')
    train_x = df_train.loc[train_ids]
    test_x = df_train.loc[test_ids]

    model = lgb.LGBMRegressor(
        random_state = 42,
        verbose = - 1,
        objective = 'tweedie',
        tweedie_variance_power = 1.0,
        metrics = 'mae',
        n_estimators = 100000,
        learning_rate = 0.03,
        boost_from_average = True,
    )

    lgb_model = model.fit(train_x[features],train_x[target],
                          eval_names=['train', 'valid'],
                          eval_set=[(train_x[features], train_x[target]),(test_x[features], test_x[target])],
                          callbacks=callbacks,
                         )

    pred_val = lgb_model.predict(test_x[features])
    df_oof_pred_i = test_x[[id_col, 'Location', 'Disease', target]].copy() #
    df_oof_pred_i[f'pred_{target}'] = pred_val
    df_oof_pred.append(df_oof_pred_i)

    pred_test = lgb_model.predict(df_test[features])
    df_test_pred[f'FOLD_{fold+1}'] = pred_test

    df_fold_imp = pd.DataFrame([[i,j] for i,j in zip(lgb_model.feature_name_, lgb_model.feature_importances_)], columns=['features','importance'])
    df_imp = pd.concat([df_imp,df_fold_imp])

df_oof_pred = pd.concat(df_oof_pred)
out = np.round(np.clip(df_oof_pred[f'pred_{target}'], 0, df_oof_pred[f'pred_{target}'].max()))
mae = mean_absolute_error(df_oof_pred[target], out)
print('MAE: ', mae)

pred_test = df_test_pred[[i for i in df_test_pred.columns if i!=id_col]].apply(lambda x: hmean(x), axis=1)
out = np.round(np.clip(pred_test, 0, pred_test.max()))
df_test_pred[target] = out
submission_lgb = df_test_pred[[id_col, target]].copy()
print('Test Data Shape:', submission_lgb.shape)
display(submission_lgb.head())
submission_lgb.to_csv('submission.csv', index=False)
df_imp = df_imp.groupby(['features']).mean().sort_values('importance', ascending=False).reset_index()
print('Zero importance features:', len(df_imp[df_imp['importance']==0]))
display(df_imp.head(20))
