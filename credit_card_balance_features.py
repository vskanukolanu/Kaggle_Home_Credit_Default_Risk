from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
#from imblearn.over_sampling import SMOTE       #over sampling of minority class in imbalanced data
#from imblearn.combine import SMOTEENN          #over sampling of minority class in imbalanced data
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
from functools import partial,reduce
import seaborn as sns
import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_columns', 100)

# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
app_fe2_df= pd.read_csv(data_path+"app_df_v7.csv")
apptest_fe2_df= pd.read_csv(data_path+"app_test_v7.csv")


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
cc_bal= pd.read_csv(data_path+"credit_card_balance.csv")
cc_bal.head(2)

cc_bal['AMT_DRAWINGS_ATM_CURRENT'][cc_bal['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
cc_bal['AMT_DRAWINGS_CURRENT'][cc_bal['AMT_DRAWINGS_CURRENT'] < 0] = np.nan

cc_bal['number_of_instalments'] = cc_bal.groupby(
    by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
    'CNT_INSTALMENT_MATURE_CUM']

cc_bal['credit_card_max_loading_of_credit_limit'] = cc_bal.groupby(
    by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
    lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

features = pd.DataFrame({'SK_ID_CURR':cc_bal['SK_ID_CURR'].unique()})

group_object = cc_bal.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].agg('nunique').reset_index()
group_object.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object= cc_bal.groupby(by=['SK_ID_CURR'])['number_of_instalments'].sum().reset_index()
group_object.rename(index=str, columns={'number_of_instalments': 'credit_card_total_instalments'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

# Credit card installments per loan
features['credit_card_installments_per_loan'] = (
    features['credit_card_total_instalments'] / features['credit_card_number_of_loans'])

#F1
group_object = cc_bal.groupby(by=['SK_ID_CURR'])['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
group_object.rename(index=str, columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

#F2
group_object = cc_bal.groupby(
    by=['SK_ID_CURR'])['SK_DPD'].agg('mean').reset_index()
group_object.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

#F3
group_object = cc_bal.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

#F4
group_object = cc_bal.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'},inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

#F5
features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features['credit_card_drawings_total']

temp_list3=[app_fe2_df,features]
app_fe3_df = reduce(lambda left,right: pd.merge(left,right,how='left',on='SK_ID_CURR'), temp_list3)

temp_list4=[apptest_fe2_df,features]
apptest_fe3_df = reduce(lambda left,right: pd.merge(left,right,how='left',on='SK_ID_CURR'), temp_list4)


CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_BALANCE',
                   'AMT_CREDIT_LIMIT_ACTUAL',
                   'AMT_DRAWINGS_ATM_CURRENT',
                   'AMT_DRAWINGS_CURRENT',
                   'AMT_DRAWINGS_OTHER_CURRENT',
                   'AMT_DRAWINGS_POS_CURRENT',
                   'AMT_PAYMENT_CURRENT',
                   'CNT_DRAWINGS_ATM_CURRENT',
                   'CNT_DRAWINGS_CURRENT',
                   'CNT_DRAWINGS_OTHER_CURRENT',
                   'CNT_INSTALMENT_MATURE_CUM',
                   'MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]


groupby_aggregate_names = []
for groupby_cols, specs in tqdm(CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES):
    group_object = cc_bal.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_fe3_df = app_fe3_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

groupby_aggregate_names = []
for groupby_cols, specs in tqdm(CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES):
    group_object = cc_bal.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        apptest_fe3_df = apptest_fe3_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)


apptest_fe3_df.to_csv(data_path+"app_test_v8.csv", index = False)

app_fe3_df.to_csv(data_path+"app_df_v8.csv", index = False)
