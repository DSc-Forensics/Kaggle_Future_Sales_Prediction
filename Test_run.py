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
all_data=pd.read_parquet('Inputs/Train_ready.gzip')
out_test_items=pd.read_csv("Outdated_test_items.csv")
sub=pd.read_csv("Inputs/sample_submission.csv")
test=pd.read_csv("Inputs/test.csv")

train_speed='fast'

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

X_train = all_data.loc[all_data['date_block_num'] <  last_block].drop('target', axis=1)
X_test =  all_data.loc[all_data['date_block_num'] == last_block].drop('target', axis=1)

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
               'num_leaves': 2**8,
               'bagging_freq':1,
               'verbose':0,
               'lambda_l2':3,
               'num_iterations':niter,
               'categorical_feature':['name:item_category_id','name:shop_id','name:city',\
                                      'name:Broad_cat']}

model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train,categorical_feature=['item_category_id',\
'shop_id','city','Broad_cat']), niter)
pred_lgb = model.predict(X_test)
pred_train=model.predict(X_train)
#pred_lgb=np.exp(pred_lgb)-0.01-make_pos

pred_lgb=pred_lgb.clip(0,20)
pred_train=pred_train.clip(0,20)

print('R-squared for TRAIN - %f' % r2_score(y_train, pred_train))
print('RMSE for LightGBM - %f' % mse(y_train, pred_train,squared=False))
#print('R-squared for LightGBM - %f' % r2_score(y_test, pred_lgb))
#print('RMSE for LightGBM - %f' % mse(y_test, pred_lgb,squared=False))
sub['item_cnt_month']=pred_lgb
sub['item_cnt_month']=sub['item_cnt_month'].clip(0,20)
lgb.plot_importance(model,max_num_features=30)
lgb.plot_importance(model,max_num_features=30,importance_type='gain')

#simple_sub=pd.read_csv("Subs/slow_6rollmean_bomb_simple_sub_bigdata_clipped.csv")
sub=sub.merge(test,on='ID')

sub=pd.merge(sub,out_test_items,how='left',on=['ID','shop_id','item_id'])
sub['item_cnt_month']=np.where(sub['open']==sub['open'],0,sub['item_cnt_month'])

sub.set_index('ID')['item_cnt_month'].to_csv("Subs/fast_nomeanenc_structured_simple_sub_smalldata_clipped.csv")
