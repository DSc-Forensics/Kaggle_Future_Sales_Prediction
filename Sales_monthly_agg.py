# -*- coding: utf-8 -*-
"""
Created on Sun May  9 16:13:40 2021

@author: PC
"""

import pandas as pd
import numpy as np
import os

from itertools import product
import gc

import warnings
warnings.filterwarnings("ignore")

import time

def downcast_dtypes(df):
        '''
            Changes column types in the dataframe: 
                    
                    `float64` type to `float32`
                    `int64`   type to `int32`
        '''
        
        # Select columns to downcast
        float_cols = [c for c in df if df[c].dtype == "float64"]
        int_cols =   [c for c in df if df[c].dtype == "int64"]
        
        # Downcast
        df[float_cols] = df[float_cols].astype(np.float32)
        df[int_cols]   = df[int_cols].astype(np.int32)
        
        return df

metrics={'item_cnt_day':['mean','median'],'item_price':['mean','median'],'Revenue':['sum','mean','median']}

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

sales=pd.read_hdf("Inputs/Translated_sales_preprocessed.hdf")

index_cols = ['shop_id', 'item_id', 'date_block_num']

grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales.loc[sales['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
grid = pd.merge(grid, sales.groupby(['item_id','Broad_cat','Broad_item','item_category_id'])['date'].count().reset_index().\
drop('date',axis=1), how='left', on='item_id')

gb = sales.groupby(index_cols)['item_cnt_day'].agg(target='sum').reset_index()
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)

gb = sales.groupby(index_cols)['item_cnt_day'].agg(target_count='count').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=index_cols).fillna(0)

del grid,gb

df_tmp = sales.loc[sales['item_cnt_day'] > 0].groupby(index_cols).agg({'item_cnt_day': 'count'})
df_tmp.reset_index(inplace=True)
df_tmp = df_tmp.rename(columns={'item_cnt_day': 'item_rate_month'})
all_data = pd.merge(all_data, df_tmp, on=index_cols, how='left')
all_data['item_rate_month'].fillna(0,inplace=True)
del df_tmp

all_data = pd.merge(all_data, sales.groupby(['shop_id','shop_type','city'])['date'].count().reset_index().\
drop('date',axis=1), how='left', on='shop_id')
all_data = pd.merge(all_data, sales.groupby(['city','city_size'])['date'].count().reset_index().\
drop('date',axis=1), how='left', on='city')   
all_data=pd.merge(all_data,sales.groupby(['date_block_num','mdays','hdays'])['date'].count().reset_index().\
drop('date',axis=1),how='left',on='date_block_num')
    
#COMPUTING COUNTS AND FREQUENCIES    
gb = sales.groupby(['date_block_num','shop_id'])['item_id'].agg(shop_count='count').reset_index()
gb2 = sales.groupby('date_block_num')['item_id'].agg(total_shop_count='count').reset_index()
gb=pd.merge(gb,gb2,how='left',on='date_block_num')
gb['shop_count_freq']=gb['shop_count']/gb['total_shop_count']
all_data = pd.merge(all_data, gb[['date_block_num','shop_id','shop_count','shop_count_freq']], how='left', on=['date_block_num','shop_id']).fillna(0)

gbi = sales.groupby(['date_block_num','item_id'])['shop_id'].agg(item_count='count').reset_index()
gbi2 = sales.groupby('date_block_num')['shop_id'].agg(total_item_count='count').reset_index()
gbi=pd.merge(gbi,gbi2,how='left',on='date_block_num')
gbi['item_count_freq']=gbi['item_count']/gbi['total_item_count']
all_data = pd.merge(all_data, gbi[['date_block_num','item_id','item_count','item_count_freq']], how='left', on=['date_block_num','item_id']).fillna(0)

gbb = sales.groupby(['date_block_num','Broad_cat'])['shop_id'].agg(broadcat_count='count').reset_index()
gbb2 = sales.groupby('date_block_num')['shop_id'].agg(total_broadcat_count='count').reset_index()
gbb=pd.merge(gbb,gbb2,how='left',on='date_block_num')
gbb['broadcat_count_freq']=gbb['broadcat_count']/gbb['total_broadcat_count']
all_data = pd.merge(all_data, gbb[['date_block_num','Broad_cat','broadcat_count','broadcat_count_freq']], how='left', on=['date_block_num','Broad_cat']).fillna(0)

gbb = sales.groupby(['date_block_num','item_category_id'])['shop_id'].agg(cat_count='count').reset_index()
gbb2 = sales.groupby('date_block_num')['shop_id'].agg(total_cat_count='count').reset_index()
gbb=pd.merge(gbb,gbb2,how='left',on='date_block_num')
gbb['cat_count_freq']=gbb['cat_count']/gbb['total_cat_count']
all_data = pd.merge(all_data, gbb[['date_block_num','item_category_id','cat_count','cat_count_freq']], how='left', on=['date_block_num','item_category_id']).fillna(0)


gb = sales.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].agg(target_shop='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
#all_data = pd.merge(all_data, broadcat_mapping, how='left', on='item_category_id')

gb = sales.groupby(['shop_type', 'date_block_num'])['item_cnt_day'].agg(target_shoptype='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['shop_type', 'date_block_num']).fillna(0)

gb = sales.groupby(['city', 'date_block_num'])['item_cnt_day'].agg(target_city='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['city', 'date_block_num']).fillna(0)

gb = sales.groupby(['item_id', 'date_block_num'])['item_cnt_day'].agg(target_item='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)

# Same as above but with item category-month aggregates
gb = sales.groupby(['item_category_id', 'date_block_num'])['item_cnt_day'].agg(target_itemcat='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['item_category_id', 'date_block_num']).fillna(0)

gb = sales.groupby(['Broad_cat', 'date_block_num'])['item_cnt_day'].agg(target_broadcat='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['Broad_cat', 'date_block_num']).fillna(0)

gb = sales.groupby(['shop_id','item_category_id', 'date_block_num'])['item_cnt_day'].agg(target_shopcat='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['shop_id','item_category_id', 'date_block_num']).fillna(0)
gb = sales.groupby(['shop_id','Broad_cat', 'date_block_num'])['item_cnt_day'].agg(target_shopbroadcat='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['shop_id','Broad_cat', 'date_block_num']).fillna(0)

gb = sales.groupby(['city','item_category_id', 'date_block_num'])['item_cnt_day'].agg(target_citycat='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['city','item_category_id', 'date_block_num']).fillna(0)
gb = sales.groupby(['city','Broad_cat', 'date_block_num'])['item_cnt_day'].agg(target_citybroadcat='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['city','Broad_cat', 'date_block_num']).fillna(0)

for metric in metrics:

    # Groupby data to get shop-item-month aggregates
    
    if 'mean' in metrics[metric] and metric=='item_price':
        gb = sales.groupby(index_cols,as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_allmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=index_cols).fillna(0)
        
        # Same as above but with shop-month aggregates
        gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shopmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
        
        
        gb = sales.groupby(['city', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_citymean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['city', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['shop_type', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shoptypemean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_type', 'date_block_num']).fillna(0)
        
        # Same as above but with item-month aggregates
        gb = sales.groupby(['item_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_itemmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['Broad_item', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_broaditemmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['Broad_item', 'date_block_num']).fillna(0)
        
        # Same as above but with item category-month aggregates
        gb = sales.groupby(['item_category_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_catmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_category_id', 'date_block_num']).fillna(0)
        
        # Same as above but with item category-month aggregates
        gb = sales.groupby(['Broad_cat', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_broadcatmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['Broad_cat', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['shop_id','item_category_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shopcatmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id','item_category_id', 'date_block_num']).fillna(0)
        gb = sales.groupby(['shop_id','Broad_cat', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shopbroadcatmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id','Broad_cat', 'date_block_num']).fillna(0)
        
    elif 'mean' in metrics[metric]:
        
        #gb = sales.groupby(index_cols,as_index=False)[metric].mean()
        #gb.rename(columns={metric:metric+'_allmean'},inplace=True)
        #all_data = pd.merge(all_data, gb, how='left', on=index_cols).fillna(0)
        
        # Same as above but with shop-month aggregates
        gbs = sales.groupby(index_cols,as_index=False)[metric].sum()
        
        gbs = pd.merge(gbs, sales.groupby(['item_id','Broad_cat','Broad_item','item_category_id'])['date'].count().reset_index().\
        drop('date',axis=1), how='left', on='item_id')
        gbs = pd.merge(gbs, sales.groupby(['shop_id','shop_type','city'])['date'].count().reset_index().\
        drop('date',axis=1), how='left', on='shop_id')  
        
        gb=gbs.groupby(['shop_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shopmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
        
        
        gb=gbs.groupby(['city', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_citymeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['city', 'date_block_num']).fillna(0)
        
        gb = gbs.groupby(['shop_type', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shoptypemean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_type', 'date_block_num']).fillna(0)
        
        # Same as above but with item-month aggregates
        gb=gbs.groupby(['item_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_itemmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
        
        # Same as above but with item category-month aggregates
        gb=gbs.groupby(['item_category_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_catmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_category_id', 'date_block_num']).fillna(0)
        
        # Same as above but with item category-month aggregates
        gb=gbs.groupby(['Broad_cat', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_broadcatmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['Broad_cat', 'date_block_num']).fillna(0)
        
        gb = gbs.groupby(['shop_id','item_category_id', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shopcatmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id','item_category_id', 'date_block_num']).fillna(0)
        gb = gbs.groupby(['shop_id','Broad_cat', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_shopbroadcatmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id','Broad_cat', 'date_block_num']).fillna(0)
        
        del gbs
    
    if 'sum' in metrics[metric]:
        gb = sales.groupby(index_cols,as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_allsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=index_cols).fillna(0)
        
        # Same as above but with shop-month aggregates
        gb = sales.groupby(['shop_id', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_shopsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)
        
        
        gb = sales.groupby(['city', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_citysum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['city', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['shop_type', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_shoptypesum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_type', 'date_block_num']).fillna(0)
        
        # Same as above but with item-month aggregates
        gb = sales.groupby(['item_id', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_itemsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['Broad_item', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_broaditemsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['Broad_item', 'date_block_num']).fillna(0)
        
        # Same as above but with item category-month aggregates
        gb = sales.groupby(['item_category_id', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_catsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_category_id', 'date_block_num']).fillna(0)
        
        # Same as above but with item category-month aggregates
        gb = sales.groupby(['Broad_cat', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_broadcatsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['Broad_cat', 'date_block_num']).fillna(0)
        
        gb = sales.groupby('date_block_num',as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_dateblocksum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on='date_block_num').fillna(0)
        
        gb = sales.groupby(['shop_id','item_category_id', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_shopcatsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id','item_category_id', 'date_block_num']).fillna(0)
        gb = sales.groupby(['shop_id','Broad_cat', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_shopbroadcatsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['shop_id','Broad_cat', 'date_block_num']).fillna(0)


all_data['binned_item_price_itemmean']=pd.qcut(all_data['item_price_itemmean'],q=12)

gb=all_data.groupby(['binned_item_price_itemmean', 'date_block_num'],as_index=False)['target'].count()
gb=gb.rename(columns={'target':'freq_encoded_item_count_freq'})
all_data = pd.merge(all_data, gb, how='left', on=['binned_item_price_itemmean', 'date_block_num'])
all_data['binned_item_price_itemmean']=all_data['binned_item_price_itemmean'].cat.add_categories(0)
#all_data['target_y'].fillna(0,inplace=True)

# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)
del gb,sales

cols_to_rename = list(all_data.columns.difference(index_cols+['item_category_id','Broad_cat','Trans_item',\
'binned_item_price_itemmean','Broad_item','city','shop_type','hdays','mdays','city_size']))
shift_range = [12,9,6,3,2,1]

start=time.time()

#all_data.index=all_data['date_block_num'].astype('str')+'_'+all_data['shop_id'].astype('str')+'_'+\
#all_data['item_id'].astype('str')

cutoff=0
for month_shift in shift_range:
    train_shift = all_data[index_cols + cols_to_rename].copy()    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)
    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0) 

    print(str(month_shift)+" lags complete")
    del train_shift
    gc.collect()
    cutoff+=3
    cutoff=min(cutoff,9)
    all_data=all_data[all_data['date_block_num']>=cutoff]
    gc.collect()

print("Took {}".format(time.time()-start))

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range] or col[-2:] in [str(item) for item in shift_range]]
#all_data = all_data[all_data['date_block_num'] >=9]

for col in fit_cols:
    if '_'.join(col.split('_')[:-1])+'_roll3_mean' in all_data.columns:
        continue
    all_data['_'.join(col.split('_')[:-1])+'_roll3_mean']=(all_data['_'.join(col.split('_')[:-1])+'_1']+\
    all_data['_'.join(col.split('_')[:-1])+'_2']+all_data['_'.join(col.split('_')[:-1])+'_3'])/3.0
        
month_shift=3
start=time.time()
cols_to_rename=[col for col in all_data.columns if '_'.join(col.split('_')[-2:])=='roll3_mean']
na_cols=[]

train_shift = all_data[index_cols + cols_to_rename].copy()    
train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
all_data = all_data[all_data['date_block_num'] >=12]    
foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
train_shift = train_shift.rename(columns=foo)
all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

print(str(month_shift)+" lags complete")
    
del train_shift

for col in fit_cols:
    if '_'.join(col.split('_')[:-1])+'_roll6_mean' in all_data.columns:
        continue
    all_data['_'.join(col.split('_')[:-1])+'_roll6_mean']=all_data['_'.join(col.split('_')[:-1])+'_roll3_mean']+\
    all_data['_'.join(col.split('_')[:-1])+'_roll3_mean_lag_3']/2.0
    #na_cols.append('_'.join(col.split('_')[:-1])+'_roll3_mean_lag_3')
    na_cols.append('_'.join(col.split('_')[:-1])+'_roll6_mean')

del all_data['binned_item_price_itemmean']    
all_data.to_parquet('Inputs/Base_train.gzip',compression='gzip')
