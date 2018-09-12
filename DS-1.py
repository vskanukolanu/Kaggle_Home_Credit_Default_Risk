from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
#from sklearn.grid_search import GridSearchCV   #Perforing grid search
from imblearn.over_sampling import SMOTE       #over sampling of minority class in imbalanced data
from imblearn.combine import SMOTEENN          #over sampling of minority class in imbalanced data
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb
import gc
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook as tqdm
from scipy.stats import skew, kurtosis, iqr

from functools import partial
from sklearn.externals import joblib

from sklearn.feature_selection import RFE,RFECV
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score,average_precision_score,precision_recall_curve,precision_score

import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_columns', 100)


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
app_df= pd.read_csv(data_path+"application_train.csv")
app_test= pd.read_csv(data_path+"application_test.csv")
app_df.head(2)


print("Number of loan applications in training data:",app_df.shape[0])
print("Number of loan applications in test data:",app_test.shape[0])
print("Number of features in test data:",app_test.shape[1])
print("Number of features in training data:",(app_df.shape[1]-1))


app_df['CODE_GENDER'].replace('XNA',np.nan, inplace=True)
app_df['CODE_GENDER'].value_counts()


app_test['CODE_GENDER'].replace('XNA',np.nan, inplace=True)
app_test['CODE_GENDER'].value_counts()

app_df['ORGANIZATION_TYPE'].replace('XNA',np.nan, inplace=True)
app_test['ORGANIZATION_TYPE'].replace('XNA',np.nan, inplace=True)

app_df['DAYS_EMPLOYED'].replace(365243,np.nan, inplace=True)
app_test['DAYS_EMPLOYED'].replace(365243,np.nan, inplace=True)

app_df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
app_test['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

app_df['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
app_test['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)


app_df['annuity_income_percentage'] = app_df['AMT_ANNUITY'] / app_df['AMT_INCOME_TOTAL']
app_df['car_to_birth_ratio'] = app_df['OWN_CAR_AGE'] / app_df['DAYS_BIRTH']
app_df['car_to_employ_ratio'] = app_df['OWN_CAR_AGE'] / app_df['DAYS_EMPLOYED']
app_df['children_ratio'] = app_df['CNT_CHILDREN'] / app_df['CNT_FAM_MEMBERS']
app_df['credit_to_annuity_ratio'] = app_df['AMT_CREDIT'] / app_df['AMT_ANNUITY']
app_df['credit_to_goods_ratio'] = app_df['AMT_CREDIT'] / app_df['AMT_GOODS_PRICE']
app_df['credit_to_income_ratio'] = app_df['AMT_CREDIT'] / app_df['AMT_INCOME_TOTAL']
app_df['days_employed_percentage'] = app_df['DAYS_EMPLOYED'] / app_df['DAYS_BIRTH']
app_df['income_credit_percentage'] = app_df['AMT_INCOME_TOTAL'] / app_df['AMT_CREDIT']
app_df['income_per_child'] = app_df['AMT_INCOME_TOTAL'] / (1 + app_df['CNT_CHILDREN'])
app_df['income_per_person'] = app_df['AMT_INCOME_TOTAL'] / app_df['CNT_FAM_MEMBERS']
app_df['payment_rate'] = app_df['AMT_ANNUITY'] / app_df['AMT_CREDIT']
app_df['phone_to_birth_ratio'] = app_df['DAYS_LAST_PHONE_CHANGE'] / app_df['DAYS_BIRTH']
app_df['phone_to_employ_ratio'] = app_df['DAYS_LAST_PHONE_CHANGE'] / app_df['DAYS_EMPLOYED']

###Adding more features for application data
app_df['cnt_non_child'] = app_df['CNT_FAM_MEMBERS'] - app_df['CNT_CHILDREN']
app_df['child_to_non_child_ratio'] = app_df['CNT_CHILDREN'] / app_df['cnt_non_child']
app_df['income_per_non_child'] = app_df['AMT_INCOME_TOTAL'] / app_df['cnt_non_child']
app_df['credit_per_person'] = app_df['AMT_CREDIT'] / app_df['CNT_FAM_MEMBERS']
app_df['credit_per_child'] = app_df['AMT_CREDIT'] / (1 + app_df['CNT_CHILDREN'])
app_df['credit_per_non_child'] = app_df['AMT_CREDIT'] / app_df['cnt_non_child']
app_df['external_sources_weighted'] = app_df.EXT_SOURCE_1 * 2 + app_df.EXT_SOURCE_2 * 3 + app_df.EXT_SOURCE_3 * 4

##Adding same set of features on test
app_test['annuity_income_percentage'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']
app_test['car_to_birth_ratio'] = app_test['OWN_CAR_AGE'] / app_test['DAYS_BIRTH']
app_test['car_to_employ_ratio'] = app_test['OWN_CAR_AGE'] / app_test['DAYS_EMPLOYED']
app_test['children_ratio'] = app_test['CNT_CHILDREN'] / app_test['CNT_FAM_MEMBERS']
app_test['credit_to_annuity_ratio'] = app_test['AMT_CREDIT'] / app_test['AMT_ANNUITY']
app_test['credit_to_goods_ratio'] = app_test['AMT_CREDIT'] / app_test['AMT_GOODS_PRICE']
app_test['credit_to_income_ratio'] = app_test['AMT_CREDIT'] / app_test['AMT_INCOME_TOTAL']
app_test['days_employed_percentage'] = app_test['DAYS_EMPLOYED'] / app_test['DAYS_BIRTH']
app_test['income_credit_percentage'] = app_test['AMT_INCOME_TOTAL'] / app_test['AMT_CREDIT']
app_test['income_per_child'] = app_test['AMT_INCOME_TOTAL'] / (1 + app_test['CNT_CHILDREN'])
app_test['income_per_person'] = app_test['AMT_INCOME_TOTAL'] / app_test['CNT_FAM_MEMBERS']
app_test['payment_rate'] = app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT']
app_test['phone_to_birth_ratio'] = app_test['DAYS_LAST_PHONE_CHANGE'] / app_test['DAYS_BIRTH']
app_test['phone_to_employ_ratio'] = app_test['DAYS_LAST_PHONE_CHANGE'] / app_test['DAYS_EMPLOYED']

###Adding more features for application data
app_test['cnt_non_child'] = app_test['CNT_FAM_MEMBERS'] - app_test['CNT_CHILDREN']
app_test['child_to_non_child_ratio'] = app_test['CNT_CHILDREN'] / app_test['cnt_non_child']
app_test['income_per_non_child'] = app_test['AMT_INCOME_TOTAL'] / app_test['cnt_non_child']
app_test['credit_per_person'] = app_test['AMT_CREDIT'] / app_test['CNT_FAM_MEMBERS']
app_test['credit_per_child'] = app_test['AMT_CREDIT'] / (1 + app_test['CNT_CHILDREN'])
app_test['credit_per_non_child'] = app_test['AMT_CREDIT'] / app_test['cnt_non_child']
app_test['external_sources_weighted'] = app_test.EXT_SOURCE_1 * 2 + app_test.EXT_SOURCE_2 * 3 + app_test.EXT_SOURCE_3 * 4

# Assuming 62 is the general average age of retirement and 365.24 days in a year
app_df['retirement_age'] = (app_df['DAYS_BIRTH'] < -22645).astype(int)
app_df['retirement_age'].value_counts()

# Assuming 4 years is the tenure of long-term employment and 365.24 days in a year
app_df['longterm_employment'] = (app_df['DAYS_EMPLOYED'] < -1460).astype(int)
app_df['longterm_employment'].value_counts()

app_test['retirement_age'] = (app_test['DAYS_BIRTH'] < -22645).astype(int)
app_test['longterm_employment'] = (app_test['DAYS_EMPLOYED'] < -1460).astype(int)


app_df['is_insured'] = app_df.apply(lambda row: 1 if row['AMT_GOODS_PRICE']<row['AMT_CREDIT'] else 0 if row['AMT_GOODS_PRICE']>=row['AMT_CREDIT'] else np.nan, axis=1)
app_test['is_insured'] = app_test.apply(lambda row: 1 if row['AMT_GOODS_PRICE']<row['AMT_CREDIT'] else 0 if row['AMT_GOODS_PRICE']>=row['AMT_CREDIT'] else np.nan, axis=1)

for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian','nanmean']:
    app_df['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
        app_df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3','external_sources_weighted']], axis=1)


for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian','nanmean']:
    app_test['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
        app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3','external_sources_weighted']], axis=1)

AGGREGATION_RECIPIES = [
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                              ('AMT_CREDIT', 'max'),
                                              ('EXT_SOURCE_1', 'mean'),
                                              ('EXT_SOURCE_2', 'mean'),
                                              ('OWN_CAR_AGE', 'max'),
                                              ('OWN_CAR_AGE', 'sum')]),
    (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                            ('AMT_INCOME_TOTAL', 'mean'),
                                            ('DAYS_REGISTRATION', 'mean'),
                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                 ('CNT_CHILDREN', 'mean'),
                                                 ('DAYS_ID_PUBLISH', 'mean')]),
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                           ('EXT_SOURCE_2', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                  ('APARTMENTS_AVG', 'mean'),
                                                  ('BASEMENTAREA_AVG', 'mean'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('EXT_SOURCE_3', 'mean'),
                                                  ('NONLIVINGAREA_AVG', 'mean'),
                                                  ('OWN_CAR_AGE', 'mean'),
                                                  ('YEARS_BUILD_AVG', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER','OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                           ('CNT_CHILDREN', 'mean'),
                           ('CNT_FAM_MEMBERS', 'mean'),
                           ('DAYS_BIRTH', 'mean'),
                           ('DAYS_EMPLOYED', 'mean'),
                           ('DAYS_ID_PUBLISH', 'mean'),
                           ('DAYS_REGISTRATION', 'mean'),
                           ('EXT_SOURCE_1', 'mean'),
                           ('EXT_SOURCE_2', 'mean'),
                           ('EXT_SOURCE_3', 'mean')]),
    (['CODE_GENDER','NAME_INCOME_TYPE'],[('AMT_ANNUITY', 'mean'),
                                        ('EXT_SOURCE_1', 'mean'),
                                        ('EXT_SOURCE_2', 'mean'),
                                        ('EXT_SOURCE_3', 'mean'),
                                        ('AMT_CREDIT', 'mean'),
                                        ('AMT_INCOME_TOTAL', 'mean'),
                                        ('AMT_INCOME_TOTAL', 'max'),
                                        ('NONLIVINGAREA_AVG', 'mean'),
                                        ('OWN_CAR_AGE', 'mean'),
                                        ('YEARS_BUILD_AVG', 'mean'),
                                        ('TOTALAREA_MODE', 'mean')])
]

groupby_aggregate_names = []
for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
    group_object = app_df.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_df = app_df.merge(group_object[select].agg(agg).reset_index()
                              .rename(index=str,columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)


## For adding the aggregate features on test, 
## we need to do the aggregates using training data and then merge those to the test data
groupby_aggregate_names = []
for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
    group_object = app_df.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_test = app_test.merge(group_object[select].agg(agg).reset_index()
                              .rename(index=str,columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)


for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
    for select, agg in tqdm(specs):
        if agg in ['mean','median','max','min']:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            diff_name = '{}_diff'.format(groupby_aggregate_name)
            abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

            app_df[diff_name] = app_df[select] - app_df[groupby_aggregate_name] 
            app_df[abs_diff_name] = np.abs(app_df[select] - app_df[groupby_aggregate_name]) 


for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
    for select, agg in tqdm(specs):
        if agg in ['mean','median','max','min']:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            diff_name = '{}_diff'.format(groupby_aggregate_name)
            abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

            app_test[diff_name] = app_test[select] - app_df[groupby_aggregate_name] 
            app_test[abs_diff_name] = np.abs(app_test[select] - app_df[groupby_aggregate_name]) 



app_df.to_csv(data_path+"app_df.csv", index = False)

app_test.to_csv(data_path+"app_test.csv", index = False)

