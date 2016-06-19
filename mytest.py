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

#feature enhancement
#fig=plt.figure()
#fig.set(alpha=0.2)
#print train_data.Survived.value_counts()
#train_data.Survived.value_counts().plot(kind='bar')
#plt.show()

#train_data.loc[:,'Sex']=train_data.loc[:,'Sex'].apply(lambda x: 0 if x=='female' else 1)

#used_data=train_data.loc[:,['PassegerId','Survived','Pclass','Sex']]
#used_data=train_data.loc[:,train_data.columns[0,1,2,4,5,6,7,9,11]]
#labels=np.array(train_data.columns)[[0,1,2,4,5,6,7,9,11]]
labels=np.array(train_data.columns)[[0,1,2,4,6,7,11]]
#labels=np.array(train_data.columns)[[0,1,2,4]]
#labels=np.array(train_data.columns)[[0,1,4]]
#labels=np.array(train_data.columns)[[0,1,2,4,6]]
used_data=train_data.loc[:,labels]
used_data.info()

#remap Sex and Embarked
#print used_data.loc[0:10,'Sex']
#print used_data.loc[used_data['Embarked']==np.nan]
#print len(pd.isnnull(used_data.Embarked))
#used_data.loc[:,'Sex']=used_data.loc[:,'Sex'].apply(lambda x: 0 if x=='female' else 1)
#used_data.loc[:,'Embarked']=used_data.loc[:,'Embarked'].apply(lambda x: 0 if x=='female' else 1)
#print used_data.loc[0:5,'Sex']
#print used_data.loc[0:10,'Embarked']

#fill NOP data with average

#quit()
feature_num=len(train_data.head(1))
cross_valid_ratio=0.2
total_train_len=len(train_data.index)
cross_valid_num=int(total_train_len*cross_valid_ratio)
print "cross valid set number %d" % cross_valid_num
train_num=total_train_len-cross_valid_num

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

#print train_data.head()
#dt.calc_entropy(train_data)
test_data=used_data#.head(30)
#test_data=used_data.head(30)
print test_data.head()
dt.calc_entropy(test_data)
all_feature=test_data.columns
feature_set=list(all_feature[2:])
print "###########"
#dt.select_opt_feature(test_data,feature_set)
#dt.select_major(test_data)
tree=dt.construct_tree(test_data,feature_set)
#print tree
dt.plot_tree(tree)

