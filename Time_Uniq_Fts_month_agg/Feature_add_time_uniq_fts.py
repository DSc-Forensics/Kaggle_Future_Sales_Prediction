# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:53:25 2021

@author: PC
"""

import numpy as np
import pandas as pd
import os
import time
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from pandas.tseries.offsets import Day, MonthBegin, MonthEnd

os.chdir("C:/Users/PC/Desktop/Programming/Kaggle/1C_Sales")

#Change version according to monthly aggregation method used
all_data = pd.read_pickle("Inputs/Base_train.pkl")
gc.collect()

"""
item_infos=all_data[['date_block_num','item_id']]
item_infos=item_infos.loc[item_infos['date_block_num']>=12]
item_infos.to_parquet('Inputs/Item_infos.gzip',compression='gzip')
del item_infos
gc.collect()
"""

#all_data=all_data.loc[all_data['date_block_num']>=15]

shift_range = [12,9,6,3,2,1]
index_cols = ['shop_id', 'item_id', 'date_block_num']

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

all_data['target_ratio_1_2']=all_data['target_lag_1']/all_data['target_lag_2']
all_data['target_ratio_2_3']=all_data['target_lag_2']/all_data['target_lag_3']
rat_cols=['target_ratio_1_2','target_ratio_2_3']

gc.collect()

mean_cols=[]
for col in all_data.columns:
    if col.endswith('roll_3_mean') or col.endswith('roll_6_mean') or col.endswith('roll_12_mean'):
        mean_cols.append(col)
        
uniq_cols=[]
for col in all_data.columns:
    if 'unique' in col:
        uniq_cols.append(col)

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range] or col[-2:] in [str(item) for item in shift_range]]
#fit_cols+=rat_cols
fit_cols+=mean_cols
fit_cols+=uniq_cols


to_drop_cols = list(set(list(all_data.columns))-(set(fit_cols)|set(['shop_id', 'date_block_num','item_category_id',\
'city','city_size','shop_type','hdays','mdays','Broad_cat','target','artist_name_or_first_word','item_name_group',\
'item_id','supercategory_id'])))

all_data[rat_cols].replace([np.inf, -np.inf], np.nan, inplace=True)
#all_data[rat_cols]=all_data[rat_cols].replace(np.nan,0)
for col in rat_cols:
    all_data[col].fillna(0,inplace=True)

items=pd.read_csv("Inputs/items.csv")
items_subset = items[['item_id', 'item_name','item_category_id']]

feature_count = 25
tfidf = TfidfVectorizer(max_features=feature_count)
items_text_fts = pd.DataFrame(tfidf.fit_transform(items_subset['item_name']).toarray())

cols =items_text_fts.columns
for i in range(feature_count):
    feature_name = 'item_name_tfidf_' + str(i)
    items_subset[feature_name] = items_text_fts[cols[i]]
    
items_subset.drop('item_name',axis=1,inplace=True)
items_subset.drop('item_category_id',axis=1,inplace=True)

all_data=all_data.merge(items_subset,how='left',on='item_id')

platform_map = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 8, 10: 1, 11: 2,
    12: 3, 13: 4, 14: 5, 15: 6, 16: 7, 17: 8, 18: 1, 19: 2, 20: 3, 21: 4, 22: 5,
    23: 6, 24: 7, 25: 8, 26: 9, 27: 10, 28: 0, 29: 0, 30: 0, 31: 0, 32: 8, 33: 11,
    34: 11, 35: 3, 36: 0, 37: 12, 38: 12, 39: 12, 40: 13, 41: 13, 42: 14, 43: 15,
    44: 15, 45: 15, 46: 14, 47: 14, 48: 14, 49: 14, 50: 14, 51: 14, 52: 14, 53: 14,
    54: 8, 55: 16, 56: 16, 57: 17, 58: 18, 59: 13, 60: 16, 61: 8, 62: 8, 63: 8, 64: 8,
    65: 8, 66: 8, 67: 8, 68: 8, 69: 8, 70: 8, 71: 8, 72: 8, 73: 0, 74: 10, 75: 0,
    76: 0, 77: 0, 78: 0, 79: 8, 80: 8, 81: 8, 82: 8, 83: 8,
}
all_data['platform_id'] = all_data['item_category_id'].map(platform_map)
    
def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

sales=pd.read_hdf("Inputs/Translated_sales_preprocessed.hdf")

sales['shop_item']=sales['shop_id'].astype('str')+'_'+sales['item_id'].astype('str')
sales['shop_itemcat']=sales['shop_id'].astype('str')+'_'+sales['item_category_id'].astype('str')

items_first_sale=pd.DataFrame(columns=['item_id','first_item_sale_month'])
shops_first_sale=pd.DataFrame(columns=['shop_id','first_shop_sale_month'])
shopitems_first_sale=pd.DataFrame(columns=['shop_item','first_shopitem_sale_month'])
shopitemcats_first_sale=pd.DataFrame(columns=['shop_itemcat','first_shopitemcat_sale_month'])

items_left=sales['item_id'].unique().tolist()
shops_left=sales['shop_id'].unique().tolist()
shopitems_left=sales['shop_item'].unique().tolist()
shopitemcats_left=sales['shop_itemcat'].unique().tolist()

for month in sales['date_block_num'].unique():            
    out=intersection(items_left,sales.loc[sales['date_block_num']==month,'item_id'].unique().tolist())
    month_sales=pd.DataFrame([out,[month]*len(out)]).T
    month_sales.columns=['item_id','first_item_sale_month']
    
    items_first_sale=items_first_sale.append(month_sales)
    items_left=list(set(items_left).symmetric_difference(set(out)))
    
    
    out=intersection(shops_left,sales.loc[sales['date_block_num']==month,'shop_id'].unique().tolist())
    month_sales=pd.DataFrame([out,[month]*len(out)]).T
    month_sales.columns=['shop_id','first_shop_sale_month']
    
    shops_first_sale=shops_first_sale.append(month_sales)
    shops_left=list(set(shops_left).symmetric_difference(set(out)))
    
    
    out=intersection(shopitems_left,sales.loc[sales['date_block_num']==month,'shop_item'].unique().tolist())
    month_sales=pd.DataFrame([out,[month]*len(out)]).T
    month_sales.columns=['shop_item','first_shopitem_sale_month']
    
    shopitems_first_sale=shopitems_first_sale.append(month_sales)
    shopitems_left=list(set(shopitems_left).symmetric_difference(set(out)))
    
    out=intersection(shopitemcats_left,sales.loc[sales['date_block_num']==month,'shop_itemcat'].unique().tolist())
    month_sales=pd.DataFrame([out,[month]*len(out)]).T
    month_sales.columns=['shop_itemcat','first_shopitemcat_sale_month']
    
    shopitemcats_first_sale=shopitemcats_first_sale.append(month_sales)
    shopitemcats_left=list(set(shopitemcats_left).symmetric_difference(set(out)))
    
#shopitems_first_sale.groupby('first_shopitem_sale_month')['shop_item'].count()
#shopitemcats_first_sale.groupby('first_shopitemcat_sale_month')['shop_itemcat'].count()

del sales
gc.collect()

all_data['first_item_sale_month']=all_data['item_id'].map(items_first_sale.set_index(\
'item_id')['first_item_sale_month'].to_dict())
all_data['first_shop_sale_month']=all_data['shop_id'].map(shops_first_sale.set_index(\
'shop_id')['first_shop_sale_month'].to_dict())

all_data['shop_item']=all_data['shop_id'].astype('str')+'_'+all_data['item_id'].astype('str')
all_data['shop_itemcat']=all_data['shop_id'].astype('str')+'_'+all_data['item_category_id'].astype('str')

all_data['first_shopitemcat_sale_month']=all_data['shop_itemcat'].map(shopitemcats_first_sale.set_index(\
'shop_itemcat')['first_shopitemcat_sale_month'].to_dict())
all_data['first_shopitem_sale_month']=all_data['shop_item'].map(shopitems_first_sale.set_index(\
'shop_item')['first_shopitem_sale_month'].to_dict())

all_data['mths_since_shopitem_first']=all_data['date_block_num']-all_data['first_shopitem_sale_month']   
all_data['mths_since_shopitem_first']=np.where(all_data['mths_since_shopitem_first']<-1,-1,all_data['mths_since_shopitem_first'])
all_data['mths_since_shopitemcat_first']=all_data['date_block_num']-all_data['first_shopitemcat_sale_month']   
all_data['mths_since_shopitemcat_first']=np.where(all_data['mths_since_shopitemcat_first']<-1,-1,all_data['mths_since_shopitemcat_first'])
all_data['mths_since_item_first']=all_data['date_block_num']-all_data['first_item_sale_month']   
all_data['mths_since_item_first']=np.where(all_data['mths_since_item_first']<-1,-1,all_data['mths_since_item_first'])
all_data['mths_since_shop_first']=all_data['date_block_num']-all_data['first_shop_sale_month']   
all_data['mths_since_shop_first']=np.where(all_data['mths_since_shop_first']<-1,-1,all_data['mths_since_shop_first'])

#del all_data['mths_since_shop_first']

all_data['mths_since_shopitemcat_first'].fillna(0,inplace=True)

"""Adding some unique items features to discover data leakages"""
all_data['unique_items_month']=all_data.groupby('date_block_num')['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_group_cats_month']=all_data.groupby(['date_block_num','item_name_group','item_category_id'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artist_cats_month']=all_data.groupby(['date_block_num',"artist_name_or_first_word",'item_category_id'])['item_id'].transform(lambda x: x.nunique())

all_data['unique_items_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby('date_block_num')['item_id'].transform(lambda x: x.nunique())
all_data['unique_items_restric_month'].fillna(0,inplace=True)
all_data['unique_item_group_cats_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num','item_name_group','item_category_id'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_group_cats_restric_month'].fillna(0,inplace=True)

all_data['mths_since_shopitemcat_first'].fillna(0,inplace=True)

#all_data.drop('first_shop_sale_month',axis=1,inplace=True)
all_data.drop(['shop_item','shop_itemcat','first_shopitem_sale_month','first_shopitemcat_sale_month','first_shop_sale_month',\
'first_item_sale_month','artist_name_or_first_word','item_name_group'],axis=1,inplace=True)
#all_data.drop(['artist_name_or_first_word','item_name_group'],axis=1,inplace=True)
    
all_data['mths_since_item_first']=all_data['mths_since_item_first'].astype('int')

all_data['seasonal']=12-all_data['date_block_num']%12
all_data=all_data.drop(to_drop_cols,axis=1)

del all_data['shop_type']
del all_data['mths_since_shopitem_first']

all_data = downcast_dtypes(all_data)

all_data['city']=all_data['city'].astype('category')
all_data['city']=all_data['city'].cat.codes

#all_data['shop_type']=all_data['shop_type'].astype('category')
#all_data['shop_type']=all_data['shop_type'].cat.codes

all_data['Broad_cat']=all_data['Broad_cat'].astype('category')
all_data['Broad_cat']=all_data['Broad_cat'].cat.codes


#all_data.to_pickle("Inputs/Base_train2.pkl")

"""---------------------------------------------------------------------------"""

sales=pd.read_hdf("Inputs/Translated_sales_preprocessed.hdf")
#all_data=pd.read_parquet('Inputs/Train_ready_v4.gzip')

sales['date']=pd.to_datetime(sales['date'], format="%d.%m.%Y")
month_last_day = sales.groupby("date_block_num").date.max().rename("month_last_day")
month_last_day[~month_last_day.dt.is_month_end] = (month_last_day[~month_last_day.dt.is_month_end] + MonthEnd())

month_first_day = sales.groupby("date_block_num").date.min().rename("month_first_day")
month_first_day[~month_first_day.dt.is_month_start] = (month_first_day[~month_first_day.dt.is_month_start] - MonthBegin())

month_length = (month_last_day - month_first_day + Day()).rename("month_length")
first_shop_date = sales.groupby("shop_id").date.min().rename("first_shop_date")
first_item_date = sales.groupby("item_id").date.min().rename("first_item_date")
first_shop_item_date = (sales.groupby(["shop_id", "item_id"]).date.min().rename("first_shop_item_date"))

all_data = all_data.merge(month_first_day, left_on="date_block_num", right_index=True, how="left")
all_data = all_data.merge(month_last_day, left_on="date_block_num", right_index=True, how="left")
all_data = all_data.merge(month_length, left_on="date_block_num", right_index=True, how="left")
all_data = all_data.merge(first_shop_date, left_on="shop_id", right_index=True, how="left")
all_data = all_data.merge(first_item_date, left_on="item_id", right_index=True, how="left")
all_data = all_data.merge(first_shop_item_date, left_on=["shop_id", "item_id"], right_index=True, how="left")

all_data["shop_open_days"] = all_data["month_last_day"] - all_data["first_shop_date"] + Day()
all_data["item_first_sale_days"] = all_data["month_last_day"] - all_data["first_item_date"] + Day()
all_data["item_in_shop_days"] = (all_data[["shop_open_days", "item_first_sale_days", "month_length"]].min(axis=1).dt.days)

all_data = all_data.drop(columns="item_first_sale_days")
all_data["item_cnt_day_avg"] = all_data["target"] / all_data["item_in_shop_days"]
all_data["month_length"] = all_data["month_length"].dt.days

all_data["shop_open_days"] = all_data["month_first_day"] - all_data["first_shop_date"]
all_data["first_item_sale_days"] = all_data["month_first_day"] - all_data["first_item_date"]
all_data["first_shop_item_sale_days"] = all_data["month_first_day"] - all_data["first_shop_item_date"]
#m["first_name_group_sale_days"] = m["month_first_day"] - m["first_name_group_date"]
all_data["shop_open_days"] = all_data["shop_open_days"].dt.days.fillna(0).clip(lower=0)
all_data["first_item_sale_days"] = (all_data["first_item_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999))
all_data["first_shop_item_sale_days"] = (all_data["first_shop_item_sale_days"].dt.days.fillna(0).clip(lower=0).replace(0, 9999))

def last_sale_days(m):
    last_shop_item_dates = []
    for dbn in range(1, 35):
        lsid_temp = (sales.query(f"date_block_num<{dbn}").groupby(["shop_id", "item_id"]).date.max()
        .rename("last_shop_item_sale_date").reset_index())
        lsid_temp["date_block_num"] = dbn
        last_shop_item_dates.append(lsid_temp)

    last_shop_item_dates = pd.concat(last_shop_item_dates)
    m = m.merge(last_shop_item_dates, on=["date_block_num", "shop_id", "item_id"], how="left")

    def days_since_last_feat(m, feat_name, date_feat_name, missingval):
        m[feat_name] = (m["month_first_day"] - m[date_feat_name]).dt.days
        m.loc[m[feat_name] > 2000, feat_name] = missingval
        m.loc[m[feat_name].isna(), feat_name] = missingval
        return m

    m = days_since_last_feat(m, "last_shop_item_sale_days", "last_shop_item_sale_date", 9999)

    m = m.drop(columns=["last_shop_item_sale_date"])
    return m

all_data = last_sale_days(all_data)
# Month id feature
all_data["month"] = all_data["month_first_day"].dt.month

all_data = all_data.drop(columns=["first_day","month_first_day","month_last_day","first_shop_date","first_item_date"\
                    ,"month","item_in_shop_days","first_shop_item_date","month_length"],errors="ignore")

all_data["cat_items_proportion"] = all_data["unique_item_cats_month"] / all_data["unique_items_month"]
all_data["name_group_new_proportion_month"] = (all_data['unique_item_groups_restric_month'] / all_data['unique_items_restric_month'])
all_data = all_data.drop(columns=["item_cnt_day_avg","unique_items_month", 'unique_items_restric_month'])
#Change version according to monthly aggregation method used
all_data.to_pickle("Inputs/Base_train2_69inc.pkl")
