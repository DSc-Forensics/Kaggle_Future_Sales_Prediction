# -*- coding: utf-8 -*-
"""
Created on Wed May 12 15:32:51 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

import xgboost as xgb
from sklearn.metrics import r2_score

import gc
from sklearn.metrics import mean_squared_error as mse

import warnings
warnings.filterwarnings("ignore")

import time

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

def run_xg(all_data,sales_by_item_id,item_infos,st_mth=21,end_mth=34,train_speed='fast',num_outdated=6,\
           onehot=False,plot=False,out_dump=False,_depth=7,_subsample=0.6,niter=100,lrate=0.04,logged=False,train_ensemble=False):
    """
    if train_speed=='fast':
        num_steps=100
    else:
        num_steps=200
    """
    if 'city' in all_data.columns and 'Broad_cat' in all_data.columns:
        all_data.drop(['city','Broad_cat'],axis=1,inplace=True)
    if onehot:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc_df = pd.DataFrame(enc.fit_transform(all_data[['item_category_id','supercategory_id','platform_id','seasonal']]).toarray())
        all_data=all_data.join(enc_df)
        
        for col in enc_df:
            all_data[col]=enc_df[col]
            del enc_df[col]
        
    #all_data.drop(['shop_id','item_category_id'],axis=1,inplace=True)
    
    for last_block in range(st_mth,end_mth):
        #last_block = 33
        print(last_block)
        train = xgb.DMatrix(all_data.loc[all_data['date_block_num'] <  last_block].drop(['target','Pred_target','item_category_id'], axis=1), label=all_data.loc[all_data['date_block_num'] <  last_block, 'target'].clip(0,20).values)
        test= xgb.DMatrix(all_data.loc[all_data['date_block_num'] ==  last_block].drop(['target','Pred_target','item_category_id'], axis=1), label=all_data.loc[all_data['date_block_num'] ==  last_block, 'target'].clip(0,20).values)
        
        y_train = all_data.loc[all_data['date_block_num'] <  last_block, 'target'].clip(0,20).values
        y_test =  all_data.loc[all_data['date_block_num'] ==  last_block, 'target'].clip(0,20).values
        
        if logged:
            y_train2 = y_train.copy()
            y_test2 = y_test.copy()
            y_train = np.log(y_train+1)
            y_test =  np.log(y_test+1)
        
        param = {
        'eta': lrate,
        'nthread':8,
        'max_depth': _depth,  
        'objective': 'reg:squarederror',
        'colsample_bytree': 0.6,'subsample':_subsample}
        steps=niter
        
        model = xgb.train(param, train, steps)

        pred_lgb = model.predict(test)
        pred_train=model.predict(train)
        
        if logged:
            pred_lgb=np.exp(pred_lgb)-1
            pred_train=np.exp(pred_train)-1
        
        """
        Hyperparameter tuning
        
        regr = xgb.XGBRegressor()
        parameters = {
         "eta"    : [0.005,0.05 ] ,
         "max_depth"        : [  8, 12],
         "gamma"            : [ 0.0,  0.4 ],
         "colsample_bytree" : [ 0.3 , 0.7 ]
         }
        
        grid = GridSearchCV(regr,
                        parameters, n_jobs=3,
                        scoring='neg_root_mean_squared_error',
                        cv=3)
    
        grid.fit(X_train, y_train)
        """
        
        pred_lgb=pred_lgb.clip(0,20)
        pred_train=pred_train.clip(0,20)
        
        del train,test
        gc.collect()
        
        outdated_items = sales_by_item_id[sales_by_item_id.loc[:,str(last_block-num_outdated):str(last_block-1)].sum(axis=1)==0]
        outdated_items = outdated_items[outdated_items.loc[:,'0':str(last_block-1)].sum(axis=1)>0]
        
        test=item_infos.loc[item_infos['date_block_num']==last_block]
        test['item_cnt_month']=pred_lgb
        test.loc[test['item_id'].isin(outdated_items['item_id']),'item_cnt_month']=0.0
        pred_lgb=test['item_cnt_month']
        
        if out_dump:
            model.dump_model('dump.raw.txt')
        
        if logged:
            print('R-squared for TRAIN - %f' % r2_score(y_train2, pred_train))
            print('RMSE for TRAIN - %f' % mse(y_train2, pred_train,squared=False))
            print('R-squared for CV - %f' % r2_score(y_test2, pred_lgb))
            print('RMSE for CV - %f' % mse(y_test2, pred_lgb,squared=False))
        
        else:
            print('R-squared for TRAIN - %f' % r2_score(y_train, pred_train))
            print('RMSE for TRAIN - %f' % mse(y_train, pred_train,squared=False))
            print('R-squared for CV - %f' % r2_score(y_test, pred_lgb))
            print('RMSE for CV - %f' % mse(y_test, pred_lgb,squared=False))
        
        all_data.loc[all_data['date_block_num'] == last_block,'Pred_target']=pred_lgb.values
        del y_train,y_test
        if plot:
            xgb.plot_importance(model,max_num_features=20)
            xgb.plot_importance(model,max_num_features=20,importance_type='gain')
    return all_data.loc[(all_data['date_block_num']>=st_mth)&(all_data['date_block_num']<end_mth)]
