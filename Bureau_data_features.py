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
app_df= pd.read_csv(data_path+"app_df.csv")
app_test= pd.read_csv(data_path+"app_test.csv")
app_df.head(2)

app_df.shape,app_test.shape

# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
bur_df= pd.read_csv(data_path+"bureau.csv")

bur_df['DAYS_CREDIT_ENDDATE'][bur_df['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
bur_df['DAYS_CREDIT_UPDATE'][bur_df['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
bur_df['DAYS_ENDDATE_FACT'][bur_df['DAYS_ENDDATE_FACT'] < -40000] = np.nan
bur_df['AMT_CREDIT_SUM_LIMIT'][bur_df['AMT_CREDIT_SUM_LIMIT']<0] = np.nan

###Number of previous credits
bureau_num_pre_Credits=bur_df.groupby(['SK_ID_CURR'])['SK_ID_BUREAU'].nunique().reset_index()
bureau_num_pre_Credits.columns=['SK_ID_CURR','bur_num_credits']

### Number of credits in each currency type
bureau_num_credits_CurType=bur_df.pivot_table( index=['SK_ID_CURR'], columns=['CREDIT_CURRENCY'], values=['SK_ID_BUREAU'], aggfunc='count').reset_index()
bureau_num_credits_CurType.columns=['SK_ID_CURR','bur_num_credits_cur1','bur_num_credits_cur2','bur_num_credits_cur3','bur_num_credits_cur4']

### Number of credits in each status
bureau_num_credits_StatusType=bur_df.pivot_table( index=['SK_ID_CURR'], columns=['CREDIT_ACTIVE'], values=['SK_ID_BUREAU'], aggfunc='count').reset_index()
bureau_num_credits_StatusType.columns=['SK_ID_CURR','bur_num_credits_active','bur_num_credits_bad_debt','bur_num_credits_closed','bur_num_credits_sold']

### Number of credits in each type
bureau_num_credits_CreditType=bur_df.pivot_table( index=['SK_ID_CURR'], columns=['CREDIT_TYPE'], values=['SK_ID_BUREAU'], aggfunc='count').reset_index()
bureau_num_credits_CreditType.columns=['SK_ID_CURR','bur_num_credits_typeA','bur_num_credits_typeCar','bur_num_credits_typeCash'
                                      ,'bur_num_credits_typeConsumer','bur_num_credits_typeCard','bur_num_credits_typeIBCredit'
                                      ,'bur_num_credits_typeBD','bur_num_credits_typeShares','bur_num_credits_typeEquipment'
                                      ,'bur_num_credits_typeWorCap','bur_num_credits_typeMicro','bur_num_credits_typeMobileO'
                                      ,'bur_num_credits_typeMortgage','bur_num_credits_typeRS','bur_num_credits_typeUnknown']


### Number of different types of loans(diversity)
bureau_num_credit_types=bur_df[['SK_ID_CURR', 'CREDIT_TYPE']].groupby(by = ['SK_ID_CURR'])['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns={'CREDIT_TYPE': 'bur_num_credit_types'})

### Number of different types of currency(diversity)
bureau_num_currency_types=bur_df[['SK_ID_CURR', 'CREDIT_CURRENCY']].groupby(by = ['SK_ID_CURR'])['CREDIT_CURRENCY'].nunique().reset_index().rename(index=str, columns={'CREDIT_CURRENCY': 'bur_num_currency_types'})

### Sum of debt
bureau_sum_debt=bur_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_DEBT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_DEBT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bur_sum_debt'})

### Sum of creditlimit
bureau_sum_creditlimit=bur_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_LIMIT']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_LIMIT'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_LIMIT': 'bur_sum_creditlimit'})

### Sum of credit
bureau_sum_credit=bur_df[['SK_ID_CURR', 'AMT_CREDIT_SUM']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM': 'bur_sum_credit'})

### Sum of credit overdue
bureau_sum_overdue=bur_df[['SK_ID_CURR', 'AMT_CREDIT_SUM_OVERDUE']].groupby(by = ['SK_ID_CURR'])['AMT_CREDIT_SUM_OVERDUE'].sum().reset_index().rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bur_sum_overdue'})

### Sum of annuity
bureau_sum_annuity=bur_df[['SK_ID_CURR', 'AMT_ANNUITY']].groupby(by = ['SK_ID_CURR'])['AMT_ANNUITY'].sum().reset_index().rename(index=str, columns={'AMT_ANNUITY': 'bur_sum_annuity'})

### Sum of number of credit prolongs
bureau_sum_credit_prolongs=bur_df[['SK_ID_CURR', 'CNT_CREDIT_PROLONG']].groupby(by = ['SK_ID_CURR'])['CNT_CREDIT_PROLONG'].sum().reset_index().rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bur_sum_cnt_credit_prolongs'})

### Number of credits with previous end date in each status
temp=bur_df[bur_df['DAYS_CREDIT_ENDDATE']<0]
bureau_num_credits_past_enddate=temp.pivot_table( index=['SK_ID_CURR'], columns=['CREDIT_ACTIVE'], values=['DAYS_CREDIT_ENDDATE'], aggfunc='count').reset_index()
bureau_num_credits_past_enddate.columns=['SK_ID_CURR','bur_num_past_active','bur_num_past_bad_debt','bur_num_past_closed','bur_num_past_sold']

### Number of credits with previous end date in each status
temp=bur_df[bur_df['DAYS_CREDIT_ENDDATE']>0]
bureau_num_credits_future_enddate=temp.pivot_table( index=['SK_ID_CURR'], columns=['CREDIT_ACTIVE'], values=['DAYS_CREDIT_ENDDATE'], aggfunc='count').reset_index()
bureau_num_credits_future_enddate.columns=['SK_ID_CURR','bur_num_future_active','bur_num_future_bad_debt','bur_num_future_closed','bur_num_future_sold']

### Sum of Annuity of Credits of each status
bureau_sum_annuity_StatusType=bur_df.pivot_table( index=['SK_ID_CURR'], columns=['CREDIT_ACTIVE'], values=['AMT_ANNUITY'], aggfunc='sum').reset_index()
bureau_sum_annuity_StatusType.columns=['SK_ID_CURR','bur_sum_annuity_active','bur_sum_annuity_bad_debt','bur_sum_annuity_closed','bur_sum_annuity_sold']

### Number of credits with DPD >90
temp=bur_df[bur_df['CREDIT_DAY_OVERDUE']>90]
bureau_num_DPD_90=temp[['SK_ID_CURR', 'CREDIT_DAY_OVERDUE']].groupby(by = ['SK_ID_CURR'])['CREDIT_DAY_OVERDUE'].count().reset_index().rename(index=str, columns={'CREDIT_DAY_OVERDUE': 'num_DPD_90'})

list_ids_train = pd.DataFrame({'SK_ID_CURR':app_df['SK_ID_CURR'].unique()})
list_ids_test = pd.DataFrame({'SK_ID_CURR':app_test['SK_ID_CURR'].unique()})
list_ids=pd.concat([list_ids_train, list_ids_test], ignore_index=True)


app_bureau_list=[list_ids,bureau_num_pre_Credits,bureau_num_credits_CurType,bureau_num_credits_StatusType,bureau_num_credits_CreditType
               ,bureau_num_credit_types,bureau_num_currency_types,bureau_sum_debt,bureau_sum_creditlimit,bureau_sum_credit,bureau_sum_overdue
               ,bureau_sum_annuity,bureau_sum_credit_prolongs,bureau_num_credits_past_enddate,bureau_num_credits_future_enddate,
               bureau_sum_annuity_StatusType,bureau_num_DPD_90]
app_bureau = reduce(lambda left,right: pd.merge(left,right,how='left',on='SK_ID_CURR'), app_bureau_list)

app_df_v2=app_df.merge(app_bureau,how='left',on='SK_ID_CURR')

app_test_v2=app_test.merge(app_bureau,how='left',on='SK_ID_CURR')


### Total active annuity including current app
app_df_v2['bur_active_annuity']=(app_df_v2['bur_sum_annuity_active']+app_df_v2['AMT_ANNUITY'])
### Total Debt to Income ratio
app_df_v2['bur_debt_income_ratio']=(app_df_v2['bur_sum_debt']/app_df_v2['AMT_INCOME_TOTAL'])
### Total Annuity to Income ratio
app_df_v2['bur_annuity_income_ratio']=(app_df_v2['bur_sum_annuity']/app_df_v2['AMT_INCOME_TOTAL'])
### Total Overdue to Income ratio
app_df_v2['bur_overdue_income_ratio']=(app_df_v2['bur_sum_overdue']/app_df_v2['AMT_INCOME_TOTAL'])
### Total credit to Income ratio
app_df_v2['bur_credit_income_ratio']=(app_df_v2['bur_sum_credit']/app_df_v2['AMT_INCOME_TOTAL'])
### Total creditlimit to Income ratio
app_df_v2['bur_creditlimit_income_ratio']=(app_df_v2['bur_sum_creditlimit']/app_df_v2['AMT_INCOME_TOTAL'])
### Active annuity to income ratio
app_df_v2['bur_active_annuity_income_ratio']=(app_df_v2['bur_active_annuity']/app_df_v2['AMT_INCOME_TOTAL'])
### Proportion of credits active
app_df_v2['bur_prop_active_credits']=(app_df_v2['bur_num_credits_active']/app_df_v2['bur_num_credits'])
### Proportion of active credits with end dates in past
app_df_v2['bur_prop_active_credits_past']=(app_df_v2['bur_num_past_active']/app_df_v2['bur_num_credits'])
### Proportion of all credits with end dates in past
app_df_v2['bur_prop_credits_past']=((app_df_v2['bur_num_past_active']+app_df_v2['bur_num_past_bad_debt']+app_df_v2['bur_num_past_closed']+app_df_v2['bur_num_past_sold'])/app_df_v2['bur_num_credits'])


#### Adding same features to test
### Total active annuity including current app
app_test_v2['bur_active_annuity']=(app_test_v2['bur_sum_annuity_active']+app_test_v2['AMT_ANNUITY'])
### Total Debt to Income ratio
app_test_v2['bur_debt_income_ratio']=(app_test_v2['bur_sum_debt']/app_test_v2['AMT_INCOME_TOTAL'])
### Total Annuity to Income ratio
app_test_v2['bur_annuity_income_ratio']=(app_test_v2['bur_sum_annuity']/app_test_v2['AMT_INCOME_TOTAL'])
### Total Overdue to Income ratio
app_test_v2['bur_overdue_income_ratio']=(app_test_v2['bur_sum_overdue']/app_test_v2['AMT_INCOME_TOTAL'])
### Total credit to Income ratio
app_test_v2['bur_credit_income_ratio']=(app_test_v2['bur_sum_credit']/app_test_v2['AMT_INCOME_TOTAL'])
### Total creditlimit to Income ratio
app_test_v2['bur_creditlimit_income_ratio']=(app_test_v2['bur_sum_creditlimit']/app_test_v2['AMT_INCOME_TOTAL'])
### Active annuity to income ratio
app_test_v2['bur_active_annuity_income_ratio']=(app_test_v2['bur_active_annuity']/app_test_v2['AMT_INCOME_TOTAL'])
### Proportion of credits active
app_test_v2['bur_prop_active_credits']=(app_test_v2['bur_num_credits_active']/app_test_v2['bur_num_credits'])
### Proportion of active credits with end dates in past
app_test_v2['bur_prop_active_credits_past']=(app_test_v2['bur_num_past_active']/app_test_v2['bur_num_credits'])
### Proportion of all credits with end dates in past
app_test_v2['bur_prop_credits_past']=((app_test_v2['bur_num_past_active']+app_test_v2['bur_num_past_bad_debt']+app_test_v2['bur_num_past_closed']+app_test_v2['bur_num_past_sold'])/app_test_v2['bur_num_credits'])


## Features in the data set related to amounts
bur_df['debt_credit_ratio']=bur_df['AMT_CREDIT_SUM_DEBT']/bur_df['AMT_CREDIT_SUM']
bur_df['credit_creditlimit_ratio']=bur_df['AMT_CREDIT_SUM']/bur_df['AMT_CREDIT_SUM_LIMIT']
bur_df['debt_creditlimit_ratio']=bur_df['AMT_CREDIT_SUM_DEBT']/bur_df['AMT_CREDIT_SUM_LIMIT']
bur_df['overdue_credit_ratio']=bur_df['AMT_CREDIT_SUM_OVERDUE']/bur_df['AMT_CREDIT_SUM']
bur_df['overdue_debt_ratio']=bur_df['AMT_CREDIT_SUM_OVERDUE']/bur_df['AMT_CREDIT_SUM_DEBT']
bur_df['overdue_creditlimit_ratio']=bur_df['AMT_CREDIT_SUM_OVERDUE']/bur_df['AMT_CREDIT_SUM_LIMIT']
bur_df['overdue_annuity_ratio']=bur_df['AMT_CREDIT_SUM_OVERDUE']/bur_df['AMT_ANNUITY']
bur_df['MaxOverdue_annuity_ratio']=bur_df['AMT_CREDIT_MAX_OVERDUE']/bur_df['AMT_ANNUITY']
bur_df['sum_overdue_days_overdue_ratio']=bur_df['AMT_CREDIT_SUM_OVERDUE']/bur_df['CREDIT_DAY_OVERDUE']
bur_df['cnt_prolonged_days_started_ratio']=bur_df['CNT_CREDIT_PROLONG']/(-1*bur_df['DAYS_CREDIT'])
bur_df['span_credit']=(-1*bur_df['DAYS_CREDIT'])-(-1*bur_df['DAYS_CREDIT_ENDDATE'])
bur_df['cnt_prolonged_span_ratio']=bur_df['CNT_CREDIT_PROLONG']/bur_df['span_credit']
bur_df['sum_overdue_span_ratio']=bur_df['AMT_CREDIT_SUM_OVERDUE']/bur_df['span_credit']
bur_df['debt_span_ratio']=bur_df['AMT_CREDIT_SUM_DEBT']/bur_df['span_credit']
bur_df['DPD_past_90']=np.where(bur_df['CREDIT_DAY_OVERDUE']>90,1,0)
bur_df['DPD_btwn_60_90']=np.where((bur_df['CREDIT_DAY_OVERDUE']>60)&(bur_df['CREDIT_DAY_OVERDUE']<=90),1,0)
bur_df['DPD_btwn_30_60']=np.where((bur_df['CREDIT_DAY_OVERDUE']>30)&(bur_df['CREDIT_DAY_OVERDUE']<=60),1,0)


BUREAU_AGGREGATION_RECIPIES = [('CREDIT_TYPE', 'count'),
                               ('CREDIT_ACTIVE', 'size'),
                               ('DPD_past_90', 'sum'),
                               ('DPD_btwn_60_90', 'sum'),
                               ('DPD_btwn_30_60', 'sum')
                               ]
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_CREDIT_SUM',
                   'AMT_CREDIT_SUM_DEBT',
                   'AMT_CREDIT_SUM_LIMIT',
                   'AMT_CREDIT_SUM_OVERDUE',
                   'AMT_CREDIT_MAX_OVERDUE',
                   'CNT_CREDIT_PROLONG',
                   'CREDIT_DAY_OVERDUE',
                   'DAYS_CREDIT',
                   'DAYS_CREDIT_ENDDATE',
                   'DAYS_ENDDATE_FACT',
                   'DAYS_CREDIT_UPDATE','debt_credit_ratio','credit_creditlimit_ratio','debt_creditlimit_ratio',
                   'overdue_credit_ratio','overdue_debt_ratio','overdue_creditlimit_ratio', 'overdue_annuity_ratio',
                   'MaxOverdue_annuity_ratio','sum_overdue_days_overdue_ratio','cnt_prolonged_days_started_ratio',
                   'span_credit','cnt_prolonged_span_ratio','sum_overdue_span_ratio'
                   ]:
        BUREAU_AGGREGATION_RECIPIES.append((select, agg))
BUREAU_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_AGGREGATION_RECIPIES)]


groupby_aggregate_names = []
for groupby_cols, specs in tqdm(BUREAU_AGGREGATION_RECIPIES):
    group_object = bur_df.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_df_v2 = app_df_v2.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

groupby_aggregate_names = []
for groupby_cols, specs in tqdm(BUREAU_AGGREGATION_RECIPIES):
    group_object = bur_df.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_test_v2 = app_test_v2.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

app_df_v2.shape,app_test_v2.shape

# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
bubal_df= pd.read_csv(data_path+"bureau_balance.csv")
bubal_df.head(2)


### Number of months in each status
### Number of credits in each currency type
bubal_num_months_Status=bubal_df.pivot_table( index=['SK_ID_BUREAU'], columns=['STATUS'], values=['MONTHS_BALANCE'], aggfunc='count').reset_index()
bubal_num_months_Status.columns=['SK_ID_BUREAU','bubal_num_months_S-0','bubal_num_months_S-1','bubal_num_months_S-2','bubal_num_months_S-3','bubal_num_months_S-4'
,'bubal_num_months_S-5','bubal_num_months_S-C','bubal_num_months_S-X']

### Number of different statuses for each loan
bubal_num_different_statuses=bubal_df[['SK_ID_BUREAU', 'STATUS']].groupby(by = ['SK_ID_BUREAU'])['STATUS'].nunique().reset_index().rename(index=str, columns={'STATUS': 'bubal_num_different_statuses'})

### Number of instalment of first occurance for each status
first_month=bubal_df.groupby(['SK_ID_BUREAU'])['MONTHS_BALANCE'].min().reset_index().rename(index=str, columns={'MONTHS_BALANCE': 'first_month'})
first_month_for_status=bubal_df.pivot_table( index=['SK_ID_BUREAU'], columns=['STATUS'], values=['MONTHS_BALANCE'], aggfunc='min').reset_index()
first_month_for_status.columns=['SK_ID_BUREAU','min_month_S0','min_month_S1','min_month_S2','min_month_S3','min_month_S4','min_month_S5','min_month_SC','min_month_SX']
temp=first_month[['SK_ID_BUREAU']]
for cols in list(first_month_for_status)[1:]:
    x=first_month.merge(first_month_for_status[['SK_ID_BUREAU',cols]],how='left',on='SK_ID_BUREAU')
    x.columns=['SK_ID_BUREAU','first_month',cols]
    x[cols]=((-1*x['first_month'])+1)+x[cols]
    temp=temp.merge(x[['SK_ID_BUREAU',cols]],how='left',on='SK_ID_BUREAU')
temp.columns=['SK_ID_BUREAU','bubal_first_inst_with_S0','bubal_first_inst_with_S1','bubal_first_inst_with_S2','bubal_first_inst_with_S3','bubal_first_inst_with_S4','bubal_first_inst_with_S5','bubal_first_inst_with_SC','bubal_first_inst_with_SX']

#Merge all the features computed from Bureau_balance
bubal_feat_list=[bur_df,bubal_num_months_Status,bubal_num_different_statuses,temp]
bubal_feat = reduce(lambda left,right: pd.merge(left,right,how='left',on='SK_ID_BUREAU'), bubal_feat_list)


BUBAL_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['bubal_num_different_statuses',
                   'bubal_first_inst_with_S0',
                   'bubal_first_inst_with_S1',
                   'bubal_first_inst_with_S2',
                   'bubal_first_inst_with_S3',
                   'bubal_first_inst_with_S4',
                   'bubal_first_inst_with_S5',
                   'bubal_first_inst_with_SC',
                   'bubal_first_inst_with_SX',
                   'bubal_num_months_S-0','bubal_num_months_S-1','bubal_num_months_S-X',
'bubal_num_months_S-2','bubal_num_months_S-3','bubal_num_months_S-4','bubal_num_months_S-5','bubal_num_months_S-C']:
        BUBAL_AGGREGATION_RECIPIES.append((select, agg))
BUBAL_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUBAL_AGGREGATION_RECIPIES)]


groupby_aggregate_names = []
for groupby_cols, specs in tqdm(BUBAL_AGGREGATION_RECIPIES):
    group_object = bubal_feat.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_df_v2 = app_df_v2.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)


groupby_aggregate_names = []
for groupby_cols, specs in tqdm(BUBAL_AGGREGATION_RECIPIES):
    group_object = bubal_feat.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_test_v2 = app_test_v2.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
        

app_test_v2.to_csv(data_path+"app_test_v2.csv", index = False)

app_df_v2.to_csv(data_path+"app_df_v2.csv", index = False)

