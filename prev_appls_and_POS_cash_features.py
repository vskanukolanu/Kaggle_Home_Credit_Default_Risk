
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
from tqdm import tqdm_notebook as tqdm

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
app_df_v2= pd.read_csv(data_path+"app_df_v2.csv")
app_test_v2= pd.read_csv(data_path+"app_test_v2.csv")

# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
prev_app_df= pd.read_csv(data_path+"previous_application.csv")
prev_app_df.head(2)

prev_app_df['NAME_CONTRACT_TYPE'].replace('XNA',np.nan, inplace=True)
prev_app_df['NAME_CASH_LOAN_PURPOSE'].replace(['XNA','XAP'],np.nan, inplace=True)

prev_app_df['prev_X-sell']=np.where(prev_app_df['PRODUCT_COMBINATION'].str.contains('X-Sell',case=False),1,0)
prev_app_df['prev_POS_loan']=np.where(prev_app_df['PRODUCT_COMBINATION'].str.contains('POS',case=False),1,0)
prev_app_df['prev_Cash_loan']=np.where(prev_app_df['PRODUCT_COMBINATION'].str.contains('Cash',case=False),1,0)
prev_app_df['prev_Card_loan']=np.where(prev_app_df['PRODUCT_COMBINATION'].str.contains('Card',case=False),1,0)


prev_app_df['prev_street']=np.where(prev_app_df['PRODUCT_COMBINATION'].str.contains('Street',case=False),1,0)
prev_app_df['prev_no_interest']=np.where(prev_app_df['PRODUCT_COMBINATION'].str.contains('without interest',case=False),1,0)


prev_app_df['expected_num_prev_payments']=(-1*prev_app_df['DAYS_FIRST_DUE'])/30

prev_app_df['prev_goods_insured']=np.where(prev_app_df['AMT_CREDIT']>prev_app_df['AMT_GOODS_PRICE'],1,0)

### Binary for early/late payment on previous applications than 1ST version date
prev_app_df['prev_early_paid']=np.where(prev_app_df['DAYS_LAST_DUE_1ST_VERSION']>prev_app_df['DAYS_LAST_DUE'],1,0)
prev_app_df['prev_late_paid']=np.where(prev_app_df['DAYS_LAST_DUE_1ST_VERSION']<prev_app_df['DAYS_LAST_DUE'],1,0)

### Diff between days_last_due and days_terminations
prev_app_df['diffdays_lastdue_termination']=(prev_app_df['DAYS_TERMINATION']-prev_app_df['DAYS_LAST_DUE'])

### Diff between application amount and credit amount granted
prev_app_df['diff_amt_applied_crdited']=(prev_app_df['AMT_APPLICATION']-prev_app_df['AMT_CREDIT'])

prev_app_df['prev_no_interest'][prev_app_df['NAME_YIELD_GROUP']=='XNA'].value_counts()

### Number of previous applications
prev_num_apps=prev_app_df.groupby(['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
prev_num_apps.columns=['SK_ID_CURR','preapps_num_credits']
app_fe2_df = app_df_v2.merge(prev_num_apps,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_apps,how='left',on='SK_ID_CURR')

### Number of previous X-Sell loans
prev_num_xsell=prev_app_df.groupby(['SK_ID_CURR'])['prev_X-sell'].sum().reset_index()
prev_num_xsell.columns=['SK_ID_CURR','preapps_num_pre_xsell']
app_fe2_df = app_df_v2.merge(prev_num_xsell,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_xsell,how='left',on='SK_ID_CURR')

### Number of previous POS loans
prev_num_pos=prev_app_df.groupby(['SK_ID_CURR'])['prev_POS_loan'].sum().reset_index()
prev_num_pos.columns=['SK_ID_CURR','preapps_num_pre_pos']
app_fe2_df = app_df_v2.merge(prev_num_pos,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_pos,how='left',on='SK_ID_CURR')

### Number of previous Cash loans
prev_num_cash=prev_app_df.groupby(['SK_ID_CURR'])['prev_Cash_loan'].sum().reset_index()
prev_num_cash.columns=['SK_ID_CURR','preapps_num_pre_cash']
app_fe2_df = app_df_v2.merge(prev_num_cash,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_cash,how='left',on='SK_ID_CURR')

### Number of previous Card loans
prev_num_card=prev_app_df.groupby(['SK_ID_CURR'])['prev_Card_loan'].sum().reset_index()
prev_num_card.columns=['SK_ID_CURR','preapps_num_pre_card']
app_fe2_df = app_df_v2.merge(prev_num_card,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_card,how='left',on='SK_ID_CURR')

### Number of previous NO Interest loans
prev_num_no_interest=prev_app_df.groupby(['SK_ID_CURR'])['prev_no_interest'].sum().reset_index()
prev_num_no_interest.columns=['SK_ID_CURR','preapps_num_pre_no_interest']
app_fe2_df = app_df_v2.merge(prev_num_no_interest,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_no_interest,how='left',on='SK_ID_CURR')

### Number of previous Goods Insured loans
prev_num_goods_insured=prev_app_df.groupby(['SK_ID_CURR'])['prev_goods_insured'].sum().reset_index()
prev_num_goods_insured.columns=['SK_ID_CURR','preapps_num_pre_goods_insured']
app_fe2_df = app_df_v2.merge(prev_num_goods_insured,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_goods_insured,how='left',on='SK_ID_CURR')

### Number of previous loans closed early
prev_num_closed_early=prev_app_df.groupby(['SK_ID_CURR'])['prev_early_paid'].sum().reset_index()
prev_num_closed_early.columns=['SK_ID_CURR','preapps_num_pre_closed_early']
app_fe2_df = app_df_v2.merge(prev_num_closed_early,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_closed_early,how='left',on='SK_ID_CURR')

### Number of previous loans closed late
prev_num_closed_late=prev_app_df.groupby(['SK_ID_CURR'])['prev_late_paid'].sum().reset_index()
prev_num_closed_late.columns=['SK_ID_CURR','preapps_num_pre_closed_late']
app_fe2_df = app_df_v2.merge(prev_num_closed_late,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_closed_late,how='left',on='SK_ID_CURR')

### Number of credits of each contract type
prev_num_credits_contracttype=prev_app_df.pivot_table( index=['SK_ID_CURR'], columns=['NAME_CONTRACT_TYPE'], values=['SK_ID_PREV'], aggfunc='count').reset_index()
prev_num_credits_contracttype.columns=['SK_ID_CURR','prev_app_num_cash-loans','prev_app_num_consumer-loans','prev_app_num_revolving-loans']
app_fe2_df = app_df_v2.merge(prev_num_credits_contracttype,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_credits_contracttype,how='left',on='SK_ID_CURR')

### Number of credits of each status type
prev_num_credits_statustype=prev_app_df.pivot_table( index=['SK_ID_CURR'], columns=['NAME_CONTRACT_STATUS'], values=['SK_ID_PREV'], aggfunc='count').reset_index()
prev_num_credits_statustype.columns=['SK_ID_CURR','prev_app_num_approved','prev_app_num_canceled','prev_app_num_refused','prev_app_num_un-used']
app_fe2_df = app_df_v2.merge(prev_num_credits_statustype,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_credits_statustype,how='left',on='SK_ID_CURR')

### Number of credits for each rejection
prev_num_credits_reject_reasons=prev_app_df.pivot_table( index=['SK_ID_CURR'], columns=['CODE_REJECT_REASON'], values=['SK_ID_PREV'], aggfunc='count').reset_index()
prev_num_credits_reject_reasons.columns=['SK_ID_CURR','prev_app_num_reject_client','prev_app_num_reject_HC','prev_app_num_reject_LIMIT','prev_app_num_reject_SCO','prev_app_num_reject_SCOFR',
'prev_app_num_reject_SYSTEM','prev_app_num_reject_VERIF','prev_app_num_reject_XAP','prev_app_num_reject_XNA']
app_fe2_df = app_df_v2.merge(prev_num_credits_reject_reasons,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_credits_reject_reasons,how='left',on='SK_ID_CURR')

### Number of credits of each client(new/old/refreshed) type
prev_num_credits_client_type=prev_app_df.pivot_table( index=['SK_ID_CURR'], columns=['NAME_CLIENT_TYPE'], values=['SK_ID_PREV'], aggfunc='count').reset_index()
prev_num_credits_client_type.columns=['SK_ID_CURR','prev_num_client_New','prev_num_client_Refreshed','prev_num_client_Repeater','prev_num_client_Unknown',]
app_fe2_df = app_df_v2.merge(prev_num_credits_client_type,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_credits_client_type,how='left',on='SK_ID_CURR')

### Status(New/Repeater/Refreshed/XNA) of customner during most recent loan application
idx=prev_app_df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].transform(max) == prev_app_df['DAYS_DECISION']
x=prev_app_df[idx]
prev_most_recent_cust_status=x[['SK_ID_CURR','NAME_CLIENT_TYPE']].sort_values(by=['SK_ID_CURR'])
prev_most_recent_cust_status.columns=['SK_ID_CURR','prev_most_recent_customer_status']
app_fe2_df = app_df_v2.merge(prev_most_recent_cust_status,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_most_recent_cust_status,how='left',on='SK_ID_CURR')

### Status(Approved/Cancelled/Refused/Unused) of application during most recent loan application
idx=prev_app_df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].transform(max) == prev_app_df['DAYS_DECISION']
x=prev_app_df[idx]
prev_most_recent_decision=x[['SK_ID_CURR','NAME_CONTRACT_STATUS']].sort_values(by=['SK_ID_CURR'])
prev_most_recent_decision.columns=['SK_ID_CURR','prev_most_recent_contract_status']
app_fe2_df = app_df_v2.merge(prev_most_recent_decision,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_most_recent_decision,how='left',on='SK_ID_CURR')

### Rejection Reason of application during most recent loan application.
idx=prev_app_df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].transform(max) == prev_app_df['DAYS_DECISION']
x=prev_app_df[idx]
prev_most_recent_rejection_reason=x[['SK_ID_CURR','CODE_REJECT_REASON']].sort_values(by=['SK_ID_CURR'])
prev_most_recent_rejection_reason.columns=['SK_ID_CURR','prev_most_recent_rejection_reason']
app_fe2_df = app_df_v2.merge(prev_most_recent_rejection_reason,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_most_recent_rejection_reason,how='left',on='SK_ID_CURR')

### source of application during most recent loan application(sale Type).
idx=prev_app_df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].transform(max) == prev_app_df['DAYS_DECISION']
x=prev_app_df[idx]
prev_most_recent_sales_type=x[['SK_ID_CURR','NAME_PRODUCT_TYPE']].sort_values(by=['SK_ID_CURR'])
prev_most_recent_sales_type.columns=['SK_ID_CURR','prev_most_recent_sales_type']
app_fe2_df = app_df_v2.merge(prev_most_recent_sales_type,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_most_recent_sales_type,how='left',on='SK_ID_CURR')

### Portfolio of application during most recent loan application.
idx=prev_app_df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].transform(max) == prev_app_df['DAYS_DECISION']
x=prev_app_df[idx]
prev_most_recent_portfolio=x[['SK_ID_CURR','NAME_PORTFOLIO']].sort_values(by=['SK_ID_CURR'])
prev_most_recent_portfolio.columns=['SK_ID_CURR','prev_most_recent_portfolio']
app_fe2_df = app_df_v2.merge(prev_most_recent_portfolio,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_most_recent_portfolio,how='left',on='SK_ID_CURR')

### Portfolio of application during most recent loan application.
idx=prev_app_df.groupby(['SK_ID_CURR'])['DAYS_DECISION'].transform(max) == prev_app_df['DAYS_DECISION']
x=prev_app_df[idx]
prev_most_recent_payment_type=x[['SK_ID_CURR','NAME_PAYMENT_TYPE']].sort_values(by=['SK_ID_CURR'])
prev_most_recent_payment_type.columns=['SK_ID_CURR','prev_most_recent_payment_type']
app_fe2_df = app_df_v2.merge(prev_most_recent_payment_type,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_most_recent_payment_type,how='left',on='SK_ID_CURR')

### Number of previous apps with last due in future
temp=prev_app_df[prev_app_df['DAYS_LAST_DUE']>0]
prev_num_lastdue_future=temp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
prev_num_lastdue_future.columns=['SK_ID_CURR','prev_num_lastdue_future']
app_fe2_df = app_df_v2.merge(prev_num_lastdue_future,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_lastdue_future,how='left',on='SK_ID_CURR')

### Number of previous apps with first due in future
temp=prev_app_df[prev_app_df['DAYS_FIRST_DUE']>0]
prev_num_firstdue_future=temp.groupby(['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
prev_num_firstdue_future.columns=['SK_ID_CURR','prev_num_firstdue_future']
app_fe2_df = app_df_v2.merge(prev_num_firstdue_future,how='left',on='SK_ID_CURR')
apptest_fe2_df = app_test_v2.merge(prev_num_firstdue_future,how='left',on='SK_ID_CURR')


PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT',
                   'expected_num_prev_payments',
                   'diff_amt_applied_crdited',
                   'diffdays_lastdue_termination'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]

groupby_aggregate_names = []
for groupby_cols, specs in PREVIOUS_APPLICATION_AGGREGATION_RECIPIES:
    group_object = prev_app_df.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_fe2_df = app_fe2_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

groupby_aggregate_names = []
for groupby_cols, specs in PREVIOUS_APPLICATION_AGGREGATION_RECIPIES:
    group_object = prev_app_df.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        apptest_fe2_df = apptest_fe2_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
pos_cash_df= pd.read_csv(data_path+"POS_CASH_Balance.csv")
pos_cash_df.head(2)

POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]

groupby_aggregate_names = []
for groupby_cols, specs in POS_CASH_BALANCE_AGGREGATION_RECIPIES:
    group_object = pos_cash_df.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_fe2_df = app_fe2_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

groupby_aggregate_names = []
for groupby_cols, specs in POS_CASH_BALANCE_AGGREGATION_RECIPIES:
    group_object = pos_cash_df.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        apptest_fe2_df = apptest_fe2_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

app_fe2_df.shape,apptest_fe2_df.shape

apptest_fe2_df.to_csv(data_path+"app_test_v4.csv", index = False)

app_fe2_df.to_csv(data_path+"app_df_v4.csv", index = False)

