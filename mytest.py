#!/usr/bin/python
#print "hello world"
import csv as csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

import decision_tree as dt

train_data=pd.read_csv('train.csv')
#train_data.info()

#labels=np.array(train_data.columns)[[0,1,2,4,5,6,7,9,11]]
#labels=np.array(train_data.columns)[[0,1,2,4,6,7,11]]
labels=np.array(train_data.columns)[[0,1,2,4,6]]
#labels=np.array(train_data.columns)[[0,1,2,4,6]]
used_data=train_data.loc[:,labels]
used_data.info()
used_data.fillna('Nan',inplace=True)



feature_num=len(train_data.head(1))
cross_valid_ratio=0.2
total_train_len=len(train_data.index)
cross_valid_num=int(total_train_len*cross_valid_ratio)
print "cross valid set number %d" % cross_valid_num
train_num=total_train_len-cross_valid_num

survived_num=len(train_data[used_data['Survived']==1].values)
survived_ratio=float(survived_num)/train_num
print "survived ratio=%f" % survived_ratio

#random_set generate
basic_set=range(0,total_train_len)
basic_len=total_train_len
result_set=[]
for i in range(0,cross_valid_num):
	rand_idx=rd.randint(0,basic_len-1)
	result_set.append(basic_set[rand_idx])
	del basic_set[rand_idx]
	basic_len=basic_len-1

cross_valid_set_idx=result_set
train_set_idx=basic_set

all_feature=used_data.columns
feature_set=list(all_feature[2:])
train_data_set=used_data.iloc[train_set_idx,:]
cross_valid_set=used_data.iloc[cross_valid_set_idx,:]
#train_data_set.info()
#cross_valid_set.info()

#train_data_set=train_data_set.head(12)
train_dt=dt.construct_tree(train_data_set,feature_set)
#dt.plot_tree(train_dt)
#print train_dt

cross_valid_ref=cross_valid_set.iloc[:,1].values
cross_valid_out=[]
correct_num=0
total_num=len(cross_valid_ref)
for idx in range(total_num):
	#print idx
	data_vector=cross_valid_set.iloc[idx,:]
	result_label=dt.select_label(train_dt,data_vector)
	cross_valid_out.append(result_label)

	if cross_valid_ref[idx]==cross_valid_out[idx]:
		correct_num+=1

correct_ratio=float(correct_num)/float(total_num)
print "correct ratio:%f" % correct_ratio

