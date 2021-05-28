# -*- coding: utf-8 -*-
"""
Created on Sun May  9 22:04:00 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
import scipy.sparse 
from sklearn.metrics import r2_score

import gc
from sklearn.metrics import mean_squared_error as mse

from statsmodels.regression import linear_model

import warnings
warnings.filterwarnings("ignore")

import time

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

#Change version according to monthly aggregation method used
all_data = pd.read_pickle("Inputs/Base_train2_69inc.pkl")

#all_data = all_data.drop(columns=["item_cnt_day_avg"])
#all_data=pd.read_parquet('Inputs/Train_ready_v5.gzip')
out_test_items=pd.read_csv("Outdated_test_items.csv")
sub=pd.read_csv("Inputs/sample_submission.csv")
test=pd.read_csv("Inputs/test.csv")

#all_data["cat_items_proportion"] = all_data["unique_item_cats_month"] / all_data["unique_items_month"]
#all_data["name_group_new_proportion_month"] = (all_data['unique_item_groups_restric_month'] / all_data['unique_items_restric_month'])

#all_data = all_data.drop(columns=["unique_items_month", 'unique_items_restric_month'])

train_speed='slow'

if train_speed=='fast':
    lrate=0.04
    niter=100
elif train_speed=='slow':
    lrate=0.004
    niter=1000
else:
    lrate=0.0004
    niter=5000

last_block=34

all_data.drop(['item_id','shop_id'],axis=1,inplace=True)

#del all_data['item_id']
X_train = all_data.loc[all_data['date_block_num'] <  last_block].drop('target', axis=1)
X_test =  all_data.loc[all_data['date_block_num'] == last_block].drop('target', axis=1)

#del X_train['mths_since_shopitem_first']
#del X_test['mths_since_shopitem_first']

#del X_train['mths_since_shopitemcat_first']
#del X_test['mths_since_shopitemcat_first']

y_train = all_data.loc[all_data['date_block_num'] <  last_block, 'target'].clip(0,20).values
y_test =  all_data.loc[all_data['date_block_num'] == last_block, 'target'].clip(0,20).values

del all_data
gc.collect()

lgb_params = {'feature_fraction': 0.75,
               'metric': 'rmse',
               'nthread':8, 
               'min_data_in_leaf': 20, 
               'bagging_fraction': 0.75, 
               'learning_rate': lrate, 
               'objective': 'mse', 
               'bagging_seed': 2**7, 
               'num_leaves': 2**9,
               'bagging_freq':1,
               'verbose':0,
               'num_iterations':niter,
               'categorical_feature':['name:item_category_id','name:city',\
                                      'name:Broad_cat','name:platform_id','name:supercategory_id','name:seasonal']}

params = {
    "num_leaves": 966,
    "cat_smooth": 45.01680827234465,
    "min_child_samples": 27,
    "min_child_weight": 0.021144950289224463,
    "max_bin": 214,
    "learning_rate": 0.01,
    "subsample_for_bin": 300000,
    "min_data_in_bin": 7,
    "colsample_bytree": 0.8,
    "subsample": 0.6,
    "subsample_freq": 5,
    "n_estimators": 8000,
    'categorical_feature':['name:item_category_id','name:shop_id','name:city',\
                                      'name:Broad_cat','name:platform_id','name:supercategory_id','name:seasonal']
}    
    
model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train,categorical_feature=['item_category_id',\
'Broad_cat','city','supercategory_id','platform_id','seasonal']), niter)
    
pred_lgb = model.predict(X_test)
pred_train=model.predict(X_train)
#pred_lgb=np.exp(pred_lgb)-0.01-make_pos

pred_lgb=pred_lgb.clip(0,20)
pred_train=pred_train.clip(0,20)

print('R-squared for TRAIN - %f' % r2_score(y_train, pred_train))
print('RMSE for LightGBM - %f' % mse(y_train, pred_train,squared=False))

sub['item_cnt_month']=pred_lgb
sub['item_cnt_month']=sub['item_cnt_month'].clip(0,20)
lgb.plot_importance(model,max_num_features=20)
lgb.plot_importance(model,max_num_features=20,importance_type='gain')

sub['item_cnt_month'].mean()

#simple_sub=pd.read_csv("Subs/slow_6rollmean_bomb_simple_sub_bigdata_clipped.csv")
sub=sub.merge(test,on='ID')

sub=pd.merge(sub,out_test_items,how='left',on=['ID','shop_id','item_id'])

sub2=sub.copy()
sub2['item_cnt_month']=np.where(sub2['open']==sub2['open'],0,sub2['item_cnt_month'])

sub2['item_cnt_month'].mean()

sub2.set_index('ID')['item_cnt_month'].to_csv("Subs/slow_moretime_tfidf_rollmeaned_nobc_simple_sub_smalldata_clipped.csv")

"""
bestargs=np.argsort(model.feature_importance)
keep_cols=[]
for arg in bestargs:
    keep_cols.append(X_train.columns[arg])
    
X_train=X_train[keep_cols]
X_test=X_test[keep_cols]
"""
#x=model.feature_importance()