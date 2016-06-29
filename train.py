#!/usr/bin/python
#print "hello world"
import numpy as np
import pandas as pd

import decision_tree as dt

train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
train_data.info()

#labels=np.array(train_data.columns)[[0,1,2,4,5,6,7,9,11]]
#labels=np.array(train_data.columns)[[0,1,2,4,6,7,11]]
#labels=np.array(train_data.columns)[[0,1,2,4,6]]
labels=np.array(train_data.columns)[[0,1,2,4,6,7]]
print labels
#labels=np.array(train_data.columns)[[0,1,2,4,6]]
used_data=train_data.loc[:,labels]
#used_data.info()
used_data.fillna('Nan',inplace=True)
#labels=np.array(test_data.columns)[[0,1,3,5]]
labels=np.array(test_data.columns)[[0,1,3,5,6]]
print labels
verify_data=test_data.loc[:,labels]
#verify_data.info()

feature_num=len(train_data.head(1))

all_feature=used_data.columns
feature_set=list(all_feature[2:])
train_data_set=used_data

train_dt=dt.construct_tree(train_data_set,feature_set)

verify_passage_idx=verify_data.iloc[:,0].values
verify_label_result=[]
total_num=len(verify_passage_idx)
for idx in range(total_num):
	#print idx
	data_vector=verify_data.iloc[idx,:]
	result_label=dt.select_label(train_dt,data_vector)
	verify_label_result.append(result_label)

out_data=pd.DataFrame({'PassengerId':verify_passage_idx,'Survived':verify_label_result})
out_data.to_csv('result.csv',index=False)
