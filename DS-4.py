
# coding: utf-8

# In[1]:


from IPython import get_ipython
get_ipython().magic('reset -sf')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


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


# In[3]:


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
app_fe2_df= pd.read_csv(data_path+"app_df_v4.csv")
apptest_fe2_df= pd.read_csv(data_path+"app_test_v4.csv")


# In[4]:


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
inst_payments= pd.read_csv(data_path+"installments_payments.csv")
inst_payments.head(2)


# In[5]:


### Difference in instalment payment to actual payment made
inst_payments['diff_payment']=(inst_payments['AMT_PAYMENT']-inst_payments['AMT_INSTALMENT'])

### Difference in instalment payment date to actual payment date
inst_payments['diff_days']=((-1*inst_payments['DAYS_ENTRY_PAYMENT'])-(-1*inst_payments['DAYS_INSTALMENT']))

inst_payments['instalment_paid_late'] = (inst_payments['diff_days'] > 0).astype(int)
inst_payments['instalment_paid_over'] = (inst_payments['diff_payment'] > 0).astype(int)


# In[7]:


dummy=inst_payments.groupby(['SK_ID_CURR','SK_ID_PREV']).count().reset_index()
list_prev_apps=dummy[['SK_ID_CURR','SK_ID_PREV']]


# In[8]:


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


# In[9]:


temp_list10=[list_prev_apps,number_instalment,number_versions,first_instalment,last_instalment,first_instalment_version
           ,last_instalment_version,first_missed_instalment,number_missed_instalment,number_missed_versions,last_missed_instalment
           ,first_missed_instalment_version,last_missed_instalment_version]
inst_feats1 = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list10)


# In[10]:


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


# In[11]:


temp_list11=[list_prev_apps,number_late_payments,first_late_payment_instalment,last_late_payment_instalment,first_late_payment_version
            , last_late_payment_version,number_late_payment_versions,number_early_payments,first_early_payment_instalment,
            last_early_payment_instalment,first_early_payment_version,last_early_payment_version,number_early_payment_versions]
inst_feats2 = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list11)


# In[12]:


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


# In[13]:


temp_list12=[list_prev_apps,number_high_payments,first_high_payment_instalment,last_high_payment_instalment,first_high_payment_version
            ,last_high_payment_version,number_high_payment_versions,number_low_payments,first_low_payment_instalment,last_low_payment_instalment
            ,first_low_payment_version,last_low_payment_version,number_low_payment_versions,max_high_payment,min_high_payment
            ,sum_high_payment,mean_high_payment,max_low_payment,min_low_payment,sum_low_payment,mean_low_payment]
inst_feats3 = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list12)


# In[14]:


temp_list13=[list_prev_apps,inst_feats1,inst_feats2,inst_feats3]
instalment_feats = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list13)


# In[6]:


# read the input files and look at the top few lines #
data_path = "/Users/venkatasravankanukolanu/Documents/Data Files/home_credit_kaggle/"
prev_app_df= pd.read_csv(data_path+"previous_application.csv")
prev_app_df.head(2)


# In[17]:


temp_list14=[prev_app_df,instalment_feats]
papp_inst_df = reduce(lambda left,right: pd.merge(left,right,how='left',on=['SK_ID_CURR','SK_ID_PREV']), temp_list14)


# In[18]:


papp_inst_df['max_high_payment_credit_ratio']=papp_inst_df['max_high_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['min_high_payment_credit_ratio']=papp_inst_df['min_high_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['sum_high_payment_credit_ratio']=papp_inst_df['sum_high_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['mean_high_payment_credit_ratio']=papp_inst_df['mean_high_payment']/papp_inst_df['AMT_CREDIT']

papp_inst_df['max_low_payment_credit_ratio']=papp_inst_df['max_low_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['min_low_payment_credit_ratio']=papp_inst_df['min_low_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['sum_low_payment_credit_ratio']=papp_inst_df['sum_low_payment']/papp_inst_df['AMT_CREDIT']
papp_inst_df['mean_low_payment_credit_ratio']=papp_inst_df['mean_low_payment']/papp_inst_df['AMT_CREDIT']


# In[19]:


prev_inst_numeric_feats=list(papp_inst_df.loc[:, papp_inst_df.dtypes != np.object])[2:]


# In[20]:


PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in prev_inst_numeric_feats:
        PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES.append((select, agg))
PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_INSTALMENT_AGGREGATION_RECIPIES)]


# In[21]:


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


# In[22]:


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


# In[23]:


apptest_fe2_df.to_csv(data_path+"app_test_v5.csv", index = False)


# In[24]:


app_fe2_df.to_csv(data_path+"app_df_v5.csv", index = False)


# #### Aggregate features

# In[25]:


INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_INSTALMENT',
                   'AMT_PAYMENT',
                   'DAYS_ENTRY_PAYMENT',
                   'DAYS_INSTALMENT',
                   'NUM_INSTALMENT_NUMBER',
                   'NUM_INSTALMENT_VERSION',
                   'diff_payment',
                   'diff_days'
                   ]:
        INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES.append((select, agg))
INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES)]


# In[26]:


groupby_aggregate_names = []
for groupby_cols, specs in tqdm(INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES):
    group_object = inst_payments.groupby(groupby_cols)
    for select, agg in tqdm(specs):
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


# In[27]:


groupby_aggregate_names = []
for groupby_cols, specs in tqdm(INSTALLMENTS_PAYMENTS_AGGREGATION_RECIPIES):
    group_object = inst_payments.groupby(groupby_cols)
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
        groupby_aggregate_names.append(groupby_aggregate_name)


# #### Aggregates with Periods

# In[7]:


import logging
import os
import random
import sys
import multiprocessing as mp
from functools import reduce

import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
#import yaml
from attrdict import AttrDict


# In[8]:


def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=1000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features  = [],[]
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)
    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features

def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [],[]
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [],[]
            yield index_chunk_, group_chunk_


# In[9]:


def add_features(feature_name, aggs, features, feature_names, groupby):
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])

    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg
        
        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str,
                                                                columns={feature_name: '{}_{}'.format(feature_name,
                                                                                                      agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features


# In[10]:


installments_ = inst_payments.sample(1000)


# In[11]:


features = pd.DataFrame({'SK_ID_CURR':installments_['SK_ID_CURR'].unique()})
groupby = installments_.groupby(['SK_ID_CURR'])


# In[34]:


feature_names = []

features, feature_names = add_features('NUM_INSTALMENT_VERSION', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('diff_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_late', ['sum','mean'],
                                     features, feature_names, groupby)

features, feature_names = add_features('diff_payment', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_over', ['sum','mean'],
                                     features, feature_names, groupby)


# In[35]:


features.shape


# In[12]:


feature_names = []


# #### Aggregates from last k instalments

# In[13]:


def last_k_instalment_features(gr,periods):
    gr_ = gr
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]

        
        features = add_features_in_group(features,gr_period, 'diff_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
       
        features = add_features_in_group(features,gr_period ,'diff_payment', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
              
    
    return features


# In[14]:


func = partial(last_k_instalment_features, periods=[1,5])

g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=1, chunk_size=100).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

display(features.head())


# In[ ]:


features.shape


# In[ ]:


app_fe2_df = app_fe2_df.merge(features, on='SK_ID_CURR', how='left')
apptest_fe2_df = apptest_fe2_df.merge(features, on='SK_ID_CURR', how='left')

