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
all_data=pd.read_parquet('Inputs/Base_train.gzip')

item_infos=all_data[['date_block_num','item_id']]
item_infos=item_infos.loc[item_infos['date_block_num']>=12]
item_infos.to_parquet('Inputs/Item_infos.gzip',compression='gzip')
del item_infos
gc.collect()

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

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range] or col[-2:] in [str(item) for item in shift_range]]

month_shift=6
start=time.time()
cols_to_rename=[col for col in all_data.columns if '_'.join(col.split('_')[-2:])=='roll6_mean']
na_cols=[]

train_shift = all_data[index_cols + cols_to_rename].copy()    
train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
all_data = all_data[all_data['date_block_num'] >=18]    
foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
train_shift = train_shift.rename(columns=foo)
all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

print(str(month_shift)+" lags complete")
del train_shift
gc.collect()

for col in fit_cols:
    if 'roll' in col:
        continue
    if '_'.join(col.split('_')[:-1])+'_roll12_mean' in all_data.columns:
        continue
    all_data['_'.join(col.split('_')[:-1])+'_roll12_mean']=all_data['_'.join(col.split('_')[:-1])+'_roll6_mean']+\
    all_data['_'.join(col.split('_')[:-1])+'_roll6_mean_lag_6']/2.0
    #na_cols.append('_'.join(col.split('_')[:-1])+'_roll3_mean_lag_3')
    na_cols.append('_'.join(col.split('_')[:-1])+'_roll6_mean')
    
all_data['target_ratio_1_2']=all_data['target_lag_1']/all_data['target_lag_2']
all_data['target_ratio_2_3']=all_data['target_lag_2']/all_data['target_lag_3']
rat_cols=['target_ratio_1_2','target_ratio_2_3']

drop_cols=[]
for col in all_data.columns:
    if col.endswith('roll3_mean_lag_3') or col.endswith('roll6_mean_lag_6'):
        drop_cols.append(col)
        
all_data=all_data.drop(drop_cols,axis=1)
gc.collect()

mean_cols=[]
for col in all_data.columns:
    if col.endswith('roll3_mean') or col.endswith('roll6_mean') or col.endswith('roll12_mean'):
        mean_cols.append(col)

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range] or col[-2:] in [str(item) for item in shift_range]]
fit_cols+=rat_cols
fit_cols+=mean_cols

to_drop_cols = list(set(list(all_data.columns))-(set(fit_cols)|set(['shop_id', 'date_block_num','item_category_id',\
            'city','city_size','shop_type','hdays','mdays','Broad_cat','target'])))

all_data.replace([np.inf, -np.inf], np.nan, inplace=True)

all_data[fit_cols]=all_data[fit_cols].replace(0,np.nan)

for col in fit_cols:
    if 'dateblock' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num'])[col].transform('max'),inplace=True)
    elif 'shoptypecat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_type','item_category_id'])[col].transform('max'),inplace=True)
    elif 'shoptype' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_type'])[col].transform('max'),inplace=True)
    elif 'shopbroadcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_id','Broad_cat'])[col].transform('max'),inplace=True)
    elif 'shopcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_id','item_category_id'])[col].transform('max'),inplace=True)
    elif 'citybroadcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','city','Broad_cat'])[col].transform('max'),inplace=True)
    elif 'citycat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','city','item_category_id'])[col].transform('max'),inplace=True)
    elif 'city' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','city'])[col].transform('max'),inplace=True)    
       
    
    elif 'shop' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','shop_id'])[col].transform('max'),inplace=True)
    elif 'broadcat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','Broad_cat'])[col].transform('max'),inplace=True)
    elif 'cat' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','item_category_id'])[col].transform('max'),inplace=True)
    #elif 'all' in col:
        #all_data[col].fillna(all_data.groupby(index_cols)[col].transform('max'),inplace=True)
    elif 'broaditem' in col:
        all_data[col].fillna(all_data.groupby(['date_block_num','Broad_item'])[col].transform('max'),inplace=True) 
    elif 'item' in col and col!='Trans_item_lag_1':
        all_data[col].fillna(all_data.groupby(['date_block_num','item_id'])[col].transform('max'),inplace=True)  

all_data[fit_cols]=all_data[fit_cols].replace(np.nan,0)

def clean_item_name(string):
    # Removes bracketed terms, special characters and extra whitespace
    string = re.sub(r"\[.*?\]", "", string)
    string = re.sub(r"\(.*?\)", "", string)
    string = re.sub(r"[^A-ZА-Яa-zа-я0-9 ]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    return string

sim_thresh=65
def partialmatchgroups(items, sim_thresh=sim_thresh):
    def strip_brackets(string):
        string = re.sub(r"\(.*?\)", "", string)
        string = re.sub(r"\[.*?\]", "", string)
        return string

    items = items.copy()
    items["nc"] = items.item_name.apply(strip_brackets)
    items["ncnext"] = np.concatenate((items["nc"].to_numpy()[1:], np.array([""])))

    def partialcompare(s):
        return fuzz.partial_ratio(s["nc"], s["ncnext"])

    items["partialmatch"] = items.apply(partialcompare, axis=1)
    # Assign groups
    grp = 0
    for i in range(items.shape[0]):
        items.loc[i, "partialmatchgroup"] = grp
        if items.loc[i, "partialmatch"] < sim_thresh:
            grp += 1
    items = items.drop(columns=["nc", "ncnext", "partialmatch"])
    return items

def extract_artist(st):
    st = st.strip()
    if st.startswith("V/A"):
        artist = "V/A"
    elif st.startswith("СБ"):
        artist = "СБ"
    else:
        # Retrieves artist names using the double space or all uppercase pattern
        mus_artist_dubspace = re.compile(r".{2,}?(?=\s{2,})")
        match_dubspace = mus_artist_dubspace.match(st)
        mus_artist_capsonly = re.compile(r"^([^a-zа-я]+\s)+")
        match_capsonly = mus_artist_capsonly.match(st)
        candidates = [match_dubspace, match_capsonly]
        candidates = [m[0] for m in candidates if m is not None]
        # Sometimes one of the patterns catches some extra words so choose the shortest one
        if len(candidates):
            artist = min(candidates, key=len)
        else:
            # If neither of the previous patterns found something, use the dot-space pattern
            mus_artist_dotspace = re.compile(r".{2,}?(?=\.\s)")
            match = mus_artist_dotspace.match(st)
            if match:
                artist = match[0]
            else:
                artist = ""
    artist = artist.upper()
    artist = re.sub(r"[^A-ZА-Я ]||\bTHE\b", "", artist)
    artist = re.sub(r"\s{2,}", " ", artist)
    artist = artist.strip()
    return artist

def first_word(string):
    # This cleans the string of special characters, excess spaces and stopwords then extracts the first word
    string = re.sub(r"[^\w\s]", "", string)
    string = re.sub(r"\s{2,}", " ", string)
    tokens = string.lower().split()
    tokens = [t for t in tokens if t not in all_stopwords]
    token = tokens[0] if len(tokens) > 0 else ""
    return token

items=pd.read_csv("Inputs/items.csv")
items_subset = items[['item_id', 'item_name','item_category_id']]
items_subset["item_name_length"] = items_subset["item_name"].apply(len)
items_subset["item_name_cleaned_length"] = items_subset["item_name"].apply(clean_item_name).apply(len)

items_subset = partialmatchgroups(items_subset)
items_subset = items_subset.rename(columns={"partialmatchgroup": "item_name_group"})
items_subset = items_subset.drop(columns="partialmatchgroup", errors="ignore")

items_subset["item_name_group"] = items_subset["item_name_group"].apply(str)
items_subset["item_name_group"] = items_subset["item_name_group"].factorize()[0]

items_subset = items_subset.copy()
all_stopwords = stopwords.words("russian")
all_stopwords = all_stopwords + stopwords.words("english")

music_categories = [55, 56, 57, 58, 59, 60]
items_subset.loc[items_subset.item_category_id.isin(music_categories), "artist_name_or_first_word"] = items_subset.loc[
    items_subset.item_category_id.isin(music_categories), "item_name"
].apply(extract_artist)
items_subset.loc[items_subset["artist_name_or_first_word"] == "", "artist_name_or_first_word"] = "other music"
items_subset.loc[~items_subset.item_category_id.isin(music_categories), "artist_name_or_first_word"] = items_subset.loc[
    ~items_subset.item_category_id.isin(music_categories), "item_name"
].apply(first_word)
items_subset.loc[items_subset["artist_name_or_first_word"] == "", "artist_name_or_first_word"] = "other non-music"
items_subset["artist_name_or_first_word"] = items_subset["artist_name_or_first_word"].factorize()[0]

feature_count = 25
tfidf = TfidfVectorizer(max_features=feature_count)
items_text_fts = pd.DataFrame(tfidf.fit_transform(items_subset['item_name']).toarray())

cols =items_text_fts.columns
for i in range(feature_count):
    feature_name = 'item_name_tfidf_' + str(i)
    items_subset[feature_name] = items_text_fts[cols[i]]
    
items_subset.drop('item_name',axis=1,inplace=True)
items_subset.drop('item_category_id',axis=1,inplace=True)
all_data=pd.merge(all_data,items_subset[['item_id','item_name_group','artist_name_or_first_word']],how='left',\
                  on='item_id')

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

supercat_map = {
    0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 2, 9: 2, 10: 1, 11: 1, 12: 1,
    13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 3, 19: 3, 20: 3, 21: 3, 22: 3, 23: 3,
    24: 3, 25: 0, 26: 2, 27: 3, 28: 3, 29: 3, 30: 3, 31: 3, 32: 2, 33: 2, 34: 2,
    35: 2, 36: 2, 37: 4, 38: 4, 39: 4, 40: 4, 41: 4, 42: 5, 43: 5, 44: 5, 45: 5,
    46: 5, 47: 5, 48: 5, 49: 5, 50: 5, 51: 5, 52: 5, 53: 5, 54: 5, 55: 6, 56: 6,
    57: 6, 58: 6, 59: 6, 60: 6, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 66: 0, 67: 0,
    68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 7, 74: 7, 75: 7, 76: 7, 77: 7, 78: 7,
    79: 2, 80: 2, 81: 0, 82: 0, 83: 0
}
all_data['supercategory_id'] = all_data['item_category_id'].map(supercat_map)

cols_to_enc=['target_lag_1','target_lag_3','target_lag_12','target_lag_roll3_mean','target_lag_roll6_mean',\
'target_lag_roll12_mean','item_price_allmean_lag_12','item_price_allmean_lag_3','item_price_allmean_lag_1',\
'item_price_allmean_lag_roll12_mean','item_price_allmean_lag_roll3_mean']
    
new_cat_cols=['item_name_group', 'artist_name_or_first_word','platform_id','supercategory_id']

for cat_col in new_cat_cols:
    for col in cols_to_enc:
        all_data[cat_col+'_'+col]=all_data.groupby(['date_block_num',cat_col])[col].transform('mean')
    
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
all_data['unique_item_groups_month']=all_data.groupby(['date_block_num','item_name_group'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_group_cats_month']=all_data.groupby(['date_block_num','item_name_group','item_category_id'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artists_month']=all_data.groupby(['date_block_num',"artist_name_or_first_word"])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artist_cats_month']=all_data.groupby(['date_block_num',"artist_name_or_first_word",'item_category_id'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_cats_month']=all_data.groupby(['date_block_num',"item_category_id"])['item_id'].transform(lambda x: x.nunique())

all_data['unique_items_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby('date_block_num')['item_id'].transform(lambda x: x.nunique())
all_data['unique_items_restric_month'].fillna(0,inplace=True)
all_data['unique_item_groups_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num','item_name_group'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_groups_restric_month'].fillna(0,inplace=True)
all_data['unique_item_group_cats_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num','item_name_group','item_category_id'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_group_cats_restric_month'].fillna(0,inplace=True)
all_data['unique_item_artists_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num',"artist_name_or_first_word"])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artists_restric_month'].fillna(0,inplace=True)
all_data['unique_item_artist_cats_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num',"artist_name_or_first_word",'item_category_id'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artist_cats_restric_month'].fillna(0,inplace=True)
all_data['unique_item_cats_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num',"item_category_id"])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_cats_restric_month'].fillna(0,inplace=True)
all_data['mths_since_shopitemcat_first'].fillna(0,inplace=True)

#all_data.drop('first_shop_sale_month',axis=1,inplace=True)
all_data.drop(['shop_item','shop_itemcat','first_shopitem_sale_month','first_shopitemcat_sale_month','first_shop_sale_month',\
'first_item_sale_month','artist_name_or_first_word','item_name_group'],axis=1,inplace=True)
all_data.drop(['artist_name_or_first_word','item_name_group'],axis=1,inplace=True)
    
all_data['mths_since_item_first']=all_data['mths_since_item_first'].astype('int')

all_data['seasonal']=12-all_data['date_block_num']%12
all_data = downcast_dtypes(all_data)

all_data=all_data.drop(to_drop_cols,axis=1)

all_data['city']=all_data['city'].astype('category')
all_data['city']=all_data['city'].cat.codes

#all_data['shop_type']=all_data['shop_type'].astype('category')
#all_data['shop_type']=all_data['shop_type'].cat.codes

all_data['Broad_cat']=all_data['Broad_cat'].astype('category')
all_data['Broad_cat']=all_data['Broad_cat'].cat.codes

del all_data['shop_type']
del all_data['mths_since_shopitem_first']

"""---------------------------------------------------------------------------"""

sales=pd.read_hdf("Inputs/Translated_sales_preprocessed.hdf")

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
    
new_cols=['item_cnt_day_avg']
for col in ['item_category_id','shop_id','city','Broad_cat','supercategory_id','platform_id','item_id']:
    all_data[col+'_item_cnt_day_avg']=all_data.groupby(['date_block_num',col])['item_cnt_day_avg'].transform('mean')
    new_cols.append(col+'_item_cnt_day_avg')
    
all_data['shop_item_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','item_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_item_item_cnt_day_avg')

all_data['shop_itemcat_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','item_category_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_itemcat_item_cnt_day_avg')

all_data['shop_supercat_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','supercategory_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_supercat_item_cnt_day_avg')

all_data['city_item_item_cnt_day_avg']=all_data.groupby(['date_block_num','city','item_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_item_item_cnt_day_avg')

all_data['city_itemcat_item_cnt_day_avg']=all_data.groupby(['date_block_num','city','item_category_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_itemcat_item_cnt_day_avg')

all_data['city_supercat_item_cnt_day_avg']=all_data.groupby(['date_block_num','city','supercategory_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_supercat_item_cnt_day_avg')
    
shift_range=[3,2,1]
index_cols = ['shop_id', 'item_id', 'date_block_num']

cur_month=18
for month_shift in shift_range:
    train_shift = all_data[index_cols + new_cols].copy()    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    all_data = all_data[all_data['date_block_num'] >=(cur_month+1)] 
    train_shift = train_shift[train_shift['date_block_num'] <=34]    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in new_cols else x
    train_shift = train_shift.rename(columns=foo)
    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

    cur_month+=1
    print(str(month_shift)+" lags complete")
    del train_shift
    gc.collect()

all_data=all_data.drop(columns=new_cols)
#all_data.to_parquet('Inputs/Train_ready_v5.gzip',compression='gzip')

#all_data.to_parquet('Inputs/Train_ready_v4.gzip',compression='gzip')
