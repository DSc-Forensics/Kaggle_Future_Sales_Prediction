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

import re
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings("ignore")

import time
from sklearn.feature_extraction.text import TfidfVectorizer
from pandas.tseries.offsets import Day, MonthBegin, MonthEnd

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

def fillna_mod(all_data,fit_cols):   
    all_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    all_data[fit_cols]=all_data[fit_cols].replace(0,np.nan)
    
    for col in fit_cols:
        if 'dateblock' in col:
            all_data[col].fillna(all_data.groupby(['date_block_num'])[col].transform('max'),inplace=True)
        elif 'itemgroup' in col:
            all_data[col].fillna(all_data.groupby(['date_block_num','item_name_group'])[col].transform('max'),inplace=True)
        elif 'itemartist' in col:
            all_data[col].fillna(all_data.groupby(['date_block_num','artist_name_or_first_word'])[col].transform('max'),inplace=True)
        elif 'supercat' in col:
            all_data[col].fillna(all_data.groupby(['date_block_num','supercategory_id'])[col].transform('max'),inplace=True)
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
    return all_data

metrics={'item_cnt_day':['mean'],'item_price':['mean'],'Revenue':['sum','mean']}

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

items_subset.drop('item_name',axis=1,inplace=True)
items_subset.drop('item_category_id',axis=1,inplace=True)
all_data=pd.merge(all_data,items_subset[['item_id','item_name_group','artist_name_or_first_word']],how='left',\
                  on='item_id')
sales=pd.merge(sales,items_subset[['item_id','item_name_group','artist_name_or_first_word']],how='left',\
                  on='item_id')

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
sales['supercategory_id'] = sales['item_category_id'].map(supercat_map)

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

gbb = sales.groupby(['date_block_num','supercategory_id'])['shop_id'].agg(supercat_count='count').reset_index()
gbb2 = sales.groupby('date_block_num')['shop_id'].agg(total_supercat_count='count').reset_index()
gbb=pd.merge(gbb,gbb2,how='left',on='date_block_num')
gbb['supercat_count_freq']=gbb['supercat_count']/gbb['total_supercat_count']
all_data = pd.merge(all_data, gbb[['date_block_num','supercategory_id','supercat_count','supercat_count_freq']], how='left', on=['date_block_num','supercategory_id']).fillna(0)

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

gb = sales.groupby(['item_name_group', 'date_block_num'])['item_cnt_day'].agg(target_itemgroup='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['item_name_group', 'date_block_num']).fillna(0)

gb = sales.groupby(['artist_name_or_first_word', 'date_block_num'])['item_cnt_day'].agg(target_itemartist='sum').reset_index()
all_data = pd.merge(all_data, gb, how='left', on=['artist_name_or_first_word', 'date_block_num']).fillna(0)

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
        
        gb = sales.groupby(['item_name_group', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_itemgroupmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_name_group', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['artist_name_or_first_word', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_itemartistmean'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['artist_name_or_first_word', 'date_block_num']).fillna(0)
        
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
        
        gbs = pd.merge(gbs, sales.groupby(['item_id','Broad_cat','Broad_item','item_category_id',\
        'artist_name_or_first_word','item_name_group'])['date'].count().reset_index().\
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
        
        gb=gbs.groupby(['item_name_group', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_itemgroupmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_name_group', 'date_block_num']).fillna(0)
        
        gb=gbs.groupby(['artist_name_or_first_word', 'date_block_num'],as_index=False)[metric].mean()
        gb.rename(columns={metric:metric+'_itemartistmeanx'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['artist_name_or_first_word', 'date_block_num']).fillna(0)
        
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
        
        gb = sales.groupby(['item_name_group', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_itemgroupsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['item_name_group', 'date_block_num']).fillna(0)
        
        gb = sales.groupby(['artist_name_or_first_word', 'date_block_num'],as_index=False)[metric].sum()
        gb.rename(columns={metric:metric+'_itemartistsum'},inplace=True)
        all_data = pd.merge(all_data, gb, how='left', on=['artist_name_or_first_word', 'date_block_num']).fillna(0)
        
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
del all_data['binned_item_price_itemmean'] 
#all_data['target_y'].fillna(0,inplace=True)

def intersection(lst1, lst2):
    # Use of hybrid method
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

items_first_sale=pd.DataFrame(columns=['item_id','first_item_sale_month'])

items_left=sales['item_id'].unique().tolist()

for month in sales['date_block_num'].unique():            
    out=intersection(items_left,sales.loc[sales['date_block_num']==month,'item_id'].unique().tolist())
    month_sales=pd.DataFrame([out,[month]*len(out)]).T
    month_sales.columns=['item_id','first_item_sale_month']
    
    items_first_sale=items_first_sale.append(month_sales)
    items_left=list(set(items_left).symmetric_difference(set(out)))

all_data['first_item_sale_month']=all_data['item_id'].map(items_first_sale.set_index(\
'item_id')['first_item_sale_month'].to_dict())

all_data['mths_since_item_first']=all_data['date_block_num']-all_data['first_item_sale_month']  
all_data['mths_since_item_first']=np.where(all_data['mths_since_item_first']<-1,-1,all_data['mths_since_item_first'])

"""Adding some unique items features to discover data leakages"""
all_data['unique_item_groups_month']=all_data.groupby(['date_block_num','item_name_group'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artists_month']=all_data.groupby(['date_block_num',"artist_name_or_first_word"])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_cats_month']=all_data.groupby(['date_block_num',"item_category_id"])['item_id'].transform(lambda x: x.nunique())

all_data['unique_item_groups_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num','item_name_group'])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_groups_restric_month'].fillna(0,inplace=True)
all_data['unique_item_artists_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num',"artist_name_or_first_word"])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_artists_restric_month'].fillna(0,inplace=True)
all_data['unique_item_cats_restric_month']=all_data.loc[all_data['mths_since_item_first']==0].groupby(['date_block_num',"item_category_id"])['item_id'].transform(lambda x: x.nunique())
all_data['unique_item_cats_restric_month'].fillna(0,inplace=True)

#all_data.drop('first_shop_sale_month',axis=1,inplace=True)
all_data.drop(['first_item_sale_month','mths_since_item_first'],axis=1,inplace=True)

sales['date']=pd.to_datetime(sales['date'], format="%d.%m.%Y")
month_last_day = sales.groupby("date_block_num").date.max().rename("month_last_day")
month_last_day[~month_last_day.dt.is_month_end] = (month_last_day[~month_last_day.dt.is_month_end] + MonthEnd())

month_first_day = sales.groupby("date_block_num").date.min().rename("month_first_day")
month_first_day[~month_first_day.dt.is_month_start] = (month_first_day[~month_first_day.dt.is_month_start] - MonthBegin())

month_length = (month_last_day - month_first_day + Day()).rename("month_length")
first_shop_date = sales.groupby("shop_id").date.min().rename("first_shop_date")
first_item_date = sales.groupby("item_id").date.min().rename("first_item_date")
#first_shop_item_date = (sales.groupby(["shop_id", "item_id"]).date.min().rename("first_shop_item_date"))

all_data = all_data.merge(month_first_day, left_on="date_block_num", right_index=True, how="left")
all_data = all_data.merge(month_last_day, left_on="date_block_num", right_index=True, how="left")
all_data = all_data.merge(month_length, left_on="date_block_num", right_index=True, how="left")
all_data = all_data.merge(first_shop_date, left_on="shop_id", right_index=True, how="left")
all_data = all_data.merge(first_item_date, left_on="item_id", right_index=True, how="left")
#all_data = all_data.merge(first_shop_item_date, left_on=["shop_id", "item_id"], right_index=True, how="left")

all_data["shop_open_days"] = all_data["month_last_day"] - all_data["first_shop_date"] + Day()
all_data["item_first_sale_days"] = all_data["month_last_day"] - all_data["first_item_date"] + Day()
all_data["item_in_shop_days"] = (all_data[["shop_open_days", "item_first_sale_days", "month_length"]].min(axis=1).dt.days)

all_data["item_cnt_day_avg"] = all_data["target"] / all_data["item_in_shop_days"]

#all_data['target2']=all_data["item_cnt_day_avg"]*all_data["month_length"]
all_data = all_data.drop(columns=["item_first_sale_days","item_in_shop_days","shop_open_days",\
            "month_last_day","first_item_date","month_length",'month_first_day', 'first_shop_date'])

del gb,sales
gc.collect()

new_cols=[]

all_data['shop_item_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','item_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_item_item_cnt_day_avg')

all_data['shop_itemcat_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','item_category_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_itemcat_item_cnt_day_avg')
all_data['shop_itemgroup_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','item_name_group'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_itemgroup_item_cnt_day_avg')
all_data['shop_supercat_item_cnt_day_avg']=all_data.groupby(['date_block_num','shop_id','supercategory_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('shop_supercat_item_cnt_day_avg')

all_data['city_item_item_cnt_day_avg']=all_data.groupby(['date_block_num','city','item_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_item_item_cnt_day_avg')
all_data['city_item_artist_cnt_day_avg']=all_data.groupby(['date_block_num','city','artist_name_or_first_word'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_item_artist_cnt_day_avg')
all_data['city_itemcat_item_cnt_day_avg']=all_data.groupby(['date_block_num','city','item_category_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_itemcat_item_cnt_day_avg')

all_data['city_supercat_item_cnt_day_avg']=all_data.groupby(['date_block_num','city','supercategory_id'])['item_cnt_day_avg'].transform('mean')
new_cols.append('city_supercat_item_cnt_day_avg')
# Downcast dtypes from 64 to 32 bit to save memory
all_data = downcast_dtypes(all_data)

#del all_data['binned_item_price_itemmean'] 

cols_to_rename = list(all_data.columns.difference(index_cols+['item_category_id','Broad_cat','Trans_item',\
'Broad_item','city','shop_type','hdays','mdays','city_size',\
'artist_name_or_first_word','item_name_group','supercategory_id']))
#shift_range = [12,9,6,3,2,1]
shift_range = [i for i in range(12,0,-1)]
lags_to_keep=[1,2,3,6,9,12]

start=time.time()

#all_data.index=all_data['date_block_num'].astype('str')+'_'+all_data['shop_id'].astype('str')+'_'+\
#all_data['item_id'].astype('str')

for month_shift in [3,6,12]:
    all_data[[i for i in range(len(cols_to_rename))]]=0.0
    new_colnames=[]
    for i in range(len(cols_to_rename)):
        new_colnames.append(cols_to_rename[i]+'_roll_'+str(month_shift)+'_mean')
    all_data.columns=all_data.columns[:len(all_data.columns)-len(cols_to_rename)].tolist()+new_colnames
    
cutoff=1
for month_shift in shift_range:
    train_shift = all_data[index_cols + cols_to_rename].copy()    
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift    
    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=foo)
    
    #all_data=all_data[all_data['date_block_num']>=cutoff]
    gc.collect()
    train_shift=train_shift[train_shift['date_block_num']<=34]
    gc.collect()
    
    for_merge=all_data[index_cols+['item_category_id','Broad_cat',\
    'Broad_item','city','shop_type','artist_name_or_first_word','item_name_group','supercategory_id']]
    for_merge = pd.merge(for_merge, train_shift, on=index_cols, how='left').fillna(0) 
    
    del train_shift
    gc.collect()
    
    for_merge=fillna_mod(for_merge,for_merge.columns[11:])
    gc.collect()
    
    if month_shift in lags_to_keep:
        all_data[for_merge.columns[11:]]=for_merge[for_merge.columns[11:]].values
    
    for month in [3,6,12]:
        new_colnames=[]
        for i in range(len(cols_to_rename)):
            new_colnames.append(cols_to_rename[i]+'_roll_'+str(month)+'_mean')
        i=0
        for col in new_colnames:
            all_data[col]=all_data[col].add(for_merge[for_merge.columns[11+i]].values)
            i+=1
        #all_data[new_colnames]=all_data[new_colnames].add(for_merge[for_merge.columns[11:]].values)
        gc.collect()
    
    #all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0) 

    print(str(month_shift)+" lags complete")
    #del train_shift
    del for_merge
    gc.collect()
    cutoff+=1
    #cutoff=min(cutoff,9)    
    gc.collect()
    
print("Took {}".format(time.time()-start))

for month in [3,6,12]:
    new_colnames=[]
    for i in range(len(cols_to_rename)):
        new_colnames.append(cols_to_rename[i]+'_roll_'+str(month)+'_mean')
    for col in new_colnames:
        all_data[col]=all_data[col]/month
    gc.collect()
    
#all_data=all_data.loc[all_data['date_block_num']>=18]
f=all_data['date_block_num']
for row in f.loc[f<=18].index:
    all_data.drop(row, inplace=True, axis=0)
#all_data = pd.read_pickle("Inputs/Base_train.pkl")
   
#all_data.to_parquet('Inputs/Base_train_v4.gzip',compression='gzip')
all_data.to_pickle("Inputs/Base_train.pkl")
#all_data.to_pickle("Inputs/Base_train_69inc.pkl")
