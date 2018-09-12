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
app_fe2_df= pd.read_csv(data_path+"app_df_v4.csv")
apptest_fe2_df= pd.read_csv(data_path+"app_test_v4.csv")

# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
inst_payments= pd.read_csv(data_path+"installments_payments.csv")
inst_payments.head(2)

### Difference in instalment payment to actual payment made
inst_payments['diff_payment']=(inst_payments['AMT_PAYMENT']-inst_payments['AMT_INSTALMENT'])

### Difference in instalment payment date to actual payment date
inst_payments['diff_days']=((-1*inst_payments['DAYS_ENTRY_PAYMENT'])-(-1*inst_payments['DAYS_INSTALMENT']))

inst_payments['instalment_paid_late'] = (inst_payments['diff_days'] > 0).astype(int)
inst_payments['instalment_paid_over'] = (inst_payments['diff_payment'] > 0).astype(int)


dummy=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV']).count().reset_index()
list_prev_apps=dummy[['SK_ID_CURR','SK_ID_PREV']]


### Number, First and Latest Instalments, Versions
number_instalment=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].nunique().reset_index()
number_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'number_instalment'},inplace=True)

number_versions=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique().reset_index()
number_versions.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'number_versiona'},inplace=True)

first_instalment=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].min().reset_index()
first_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'first_instalment'},inplace=True)

last_instalment=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
last_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'last_instalment'},inplace=True)

first_instalment_version=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].min().reset_index()
first_instalment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'first_instalment_version'},inplace=True)

last_instalment_version=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().reset_index()
last_instalment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'last_instalment_version'},inplace=True)



### Number, First and Latest Missed Instalments, Versions 
test1=inst_payments[inst_payments['AMT_PAYMENT']==0]
first_missed_instalment=test1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].min().reset_index()
first_missed_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'first_missed_instalment'},inplace=True)

number_missed_instalment=test1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].nunique().reset_index()
number_missed_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'number_missed_instalment'},inplace=True)

number_missed_versions=test1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique().reset_index()
number_missed_versions.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'number_missed_versions'},inplace=True)

last_missed_instalment=test1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
last_missed_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'last_missed_instalment'},inplace=True)

first_missed_instalment_version=test1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].min().reset_index()
first_missed_instalment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'first_missed_instalment_version'},inplace=True)

last_missed_instalment_version=test1.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().reset_index()
last_missed_instalment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'last_missed_instalment_version'},inplace=True)


temp_list10=[list_prev_apps,number_instalment,number_versions,first_instalment,last_instalment,first_instalment_version
           ,last_instalment_version,first_missed_instalment,number_missed_instalment,number_missed_versions,last_missed_instalment
           ,first_missed_instalment_version,last_missed_instalment_version]
inst_feats1 = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list10)


### Number, First and Latest late payment
test2=inst_payments[(inst_payments['diff_days']<0) & (inst_payments['AMT_PAYMENT']!=0)]
number_late_payments=test2.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_days'].count().reset_index()
number_late_payments.rename(index=str, columns={'diff_days':'number_late_payments'},inplace=True)

first_late_payment_instalment=test2.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].min().reset_index()
first_late_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'first_late_payment_instalment'},inplace=True)

last_late_payment_instalment=test2.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
last_late_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'last_late_payment_instalment'},inplace=True)

first_late_payment_version=test2.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].min().reset_index()
first_late_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'first_late_payment_version'},inplace=True)

last_late_payment_version=test2.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().reset_index()
last_late_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'last_late_payment_version'},inplace=True)


number_late_payment_versions=test2.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique().reset_index()
number_late_payment_versions.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'number_late_payment_versions'},inplace=True)



### Number, First and Latest early payment
test3=inst_payments[(inst_payments['diff_days']>0) & (inst_payments['AMT_PAYMENT']!=0)]
number_early_payments=test3.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_days'].count().reset_index()
number_early_payments.rename(index=str, columns={'diff_days':'number_early_payments'},inplace=True)

first_early_payment_instalment=test3.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].min().reset_index()
first_early_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'first_early_payment_instalment'},inplace=True)

last_early_payment_instalment=test3.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
last_early_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'last_early_payment_instalment'},inplace=True)

first_early_payment_version=test3.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].min().reset_index()
first_early_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'first_early_payment_version'},inplace=True)

last_early_payment_version=test3.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().reset_index()
last_early_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'last_early_payment_version'},inplace=True)


number_early_payment_versions=test3.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique().reset_index()
number_early_payment_versions.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'number_early_payment_versions'},inplace=True)


temp_list11=[list_prev_apps,number_late_payments,first_late_payment_instalment,last_late_payment_instalment,first_late_payment_version
            , last_late_payment_version,number_late_payment_versions,number_early_payments,first_early_payment_instalment,
            last_early_payment_instalment,first_early_payment_version,last_early_payment_version,number_early_payment_versions]
inst_feats2 = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list11)


### Number, First and Latest high payments
test4=inst_payments[(inst_payments['diff_payment']>0) & (inst_payments['AMT_PAYMENT']!=0)]
number_high_payments=test4.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].count().reset_index()
number_high_payments.rename(index=str, columns={'diff_payment':'number_high_payments'},inplace=True)

first_high_payment_instalment=test4.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].min().reset_index()
first_high_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'first_high_payment_instalment'},inplace=True)

last_high_payment_instalment=test4.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
last_high_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'last_high_payment_instalment'},inplace=True)

first_high_payment_version=test4.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].min().reset_index()
first_high_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'first_high_payment_version'},inplace=True)

last_high_payment_version=test4.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().reset_index()
last_high_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'last_high_payment_version'},inplace=True)

number_high_payment_versions=test4.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique().reset_index()
number_high_payment_versions.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'number_high_payment_versions'},inplace=True)

                                                
### Number, First and Latest low payments
test5=inst_payments[(inst_payments['diff_payment']<0) & (inst_payments['AMT_PAYMENT']!=0)]
number_low_payments=test5.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].count().reset_index()
number_low_payments.rename(index=str, columns={'diff_payment':'number_low_payments'},inplace=True)

first_low_payment_instalment=test5.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].min().reset_index()
first_low_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'first_low_payment_instalment'},inplace=True)

last_low_payment_instalment=test5.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
last_low_payment_instalment.rename(index=str, columns={'NUM_INSTALMENT_NUMBER':'last_low_payment_instalment'},inplace=True)

first_low_payment_version=test5.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].min().reset_index()
first_low_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'first_low_payment_version'},inplace=True)

last_low_payment_version=test5.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].max().reset_index()
last_low_payment_version.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'last_low_payment_version'},inplace=True)

number_low_payment_versions=test5.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique().reset_index()
number_low_payment_versions.rename(index=str, columns={'NUM_INSTALMENT_VERSION':'number_low_payment_versions'},inplace=True)

#####Some extra features
test6=inst_payments[(inst_payments['diff_payment']>0) & (inst_payments['AMT_PAYMENT']!=0)]
max_high_payment=test6.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].max().reset_index()
max_high_payment.rename(index=str, columns={'diff_payment':'max_high_payment'},inplace=True)

min_high_payment=test6.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].min().reset_index()
min_high_payment.rename(index=str, columns={'diff_payment':'min_high_payment'},inplace=True)

sum_high_payment=test6.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].sum().reset_index()
sum_high_payment.rename(index=str, columns={'diff_payment':'sum_high_payment'},inplace=True)

mean_high_payment=test6.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].mean().reset_index()
mean_high_payment.rename(index=str, columns={'diff_payment':'mean_high_payment'},inplace=True)

test7=inst_payments[(inst_payments['diff_payment']<0) & (inst_payments['AMT_PAYMENT']!=0)]
max_low_payment=test7.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].min().reset_index()
max_low_payment.rename(index=str, columns={'diff_payment':'max_low_payment'},inplace=True)

min_low_payment=test7.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].max().reset_index()
min_low_payment.rename(index=str, columns={'diff_payment':'min_low_payment'},inplace=True)

sum_low_payment=test7.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].sum().reset_index()
sum_low_payment.rename(index=str, columns={'diff_payment':'sum_low_payment'},inplace=True)

mean_low_payment=test7.groupby(['SK_ID_CURR','SK_ID_PREV'])['diff_payment'].mean().reset_index()
mean_low_payment.rename(index=str, columns={'diff_payment':'mean_low_payment'},inplace=True)


temp_list12=[list_prev_apps,number_high_payments,first_high_payment_instalment,last_high_payment_instalment,first_high_payment_version
            ,last_high_payment_version,number_high_payment_versions,number_low_payments,first_low_payment_instalment,last_low_payment_instalment
            ,first_low_payment_version,last_low_payment_version,number_low_payment_versions,max_high_payment,min_high_payment
            ,sum_high_payment,mean_high_payment,max_low_payment,min_low_payment,sum_low_payment,mean_low_payment]
inst_feats3 = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list12)


temp_list13=[list_prev_apps,inst_feats1,inst_feats2,inst_feats3]
instalment_feats = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list13)


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
prev_app_df= pd.read_csv(data_path+"previous_application.csv")
prev_app_df.head(2)


temp_list14=[prev_app_df,instalment_feats]
papp_inst_df = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list14)


papp_inst_df['max_high_payment_credit_ratio']=papp_inst_df['max_high_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['min_high_payment_credit_ratio']=papp_inst_df['min_high_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['sum_high_payment_credit_ratio']=papp_inst_df['sum_high_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['mean_high_payment_credit_ratio']=papp_inst_df['mean_high_payment']/papp_inst_df['AMT_CREDIT']

papp_inst_df['max_low_payment_credit_ratio']=papp_inst_df['max_low_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['min_low_payment_credit_ratio']=papp_inst_df['min_low_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['sum_low_payment_credit_ratio']=papp_inst_df['sum_low_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['mean_low_payment_credit_ratio']=papp_inst_df['mean_low_payment']/papp_inst_df['AMT_CREDIT']


prev_inst_numeric_feats=list(papp_inst_df.loc[:, papp_inst_df.dtypes != np.object])[2:]

PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in prev_inst_numeric_feats:
        PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES)]

groupby_aggregate_names = []
for groupby_cols, specs in (PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES):
    group_object = papp_inst_df.groupby(groupby_cols)
    for select, agg in (specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        app_fe2_df = app_fe2_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')

groupby_aggregate_names = []
for groupby_cols, specs in tqdm(PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES):
    group_object = papp_inst_df.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        apptest_fe2_df = apptest_fe2_df.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')

apptest_fe2_df.to_csv(data_path+"app_test_v5.csv", index = False)

app_fe2_df.to_csv(data_path+"app_df_v5.csv", index = False)
