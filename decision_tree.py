#!/usr/bin/python

#decision tree using ID3 algorithm
#data set structure
#0=>sampleid
#1=>label
#2~N=>feature

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy as cp


label_name="Survived"
sample_id_idx=0
label_index=1
feature_start_idx=2
def calc_entropy(DataSet):
	label_count={}
	total_num=len(DataSet.index)
	#print "total num:%d" % total_num
	for idx in range(total_num):
		data_vector=DataSet.iloc[idx,:]
		#print data_vector
		current_label=data_vector[label_index]
		if current_label not in label_count.keys():
			label_count[current_label]=0
		label_count[current_label]+=1

	entropy=0
	for current_label in label_count:
		prob=float(label_count[current_label])/total_num
		entropy+=-1*prob*np.log2(prob)
		#print "label:%r count:%d prob %f" % (current_label,label_count[current_label],prob)
	
	#print "entropy:%f" % entropy
	return entropy

def select_opt_feature(DataSet,feature_set):
	input_entropy=calc_entropy(DataSet)
	best_feature_idx=0
	best_entropy=input_entropy
	total_num=len(DataSet.index)
	for current_feature in feature_set:
		#print "current feature:%s" % current_feature
		#split data set
		feature_val_list=set(DataSet[current_feature])
		#print feature_val_list
		current_entropy=0
		for feature_val in feature_val_list:
			subset=DataSet[DataSet[current_feature]==feature_val]
			sample_num_in_feature=len(subset.index)
			prob=float(sample_num_in_feature)/total_num
			current_entropy+=prob*calc_entropy(subset)
		if current_entropy<best_entropy:
			best_entropy=current_entropy
			best_feature_idx=feature_set.index(current_feature)
			#print "update best feature:%r feature idx:%d input entropy:%f best entropy:%f" %(feature_set[best_feature_idx],best_feature_idx,input_entropy,best_entropy)


	#print "final best feature:%r feature idx:%d input entropy:%f best entropy:%f" %(feature_set[best_feature_idx],best_feature_idx,input_entropy,best_entropy)
	#print "best feature:%s" % feature_set[best_feature_idx]
	return best_feature_idx


def select_major(leaf):
	label_count={}
	total_num=len(leaf.index)
	print "leaf count:%d" %total_num
	#if total_num==1:
	#return (leaf.iloc[0,label_index],1,0)
	for idx in range(total_num):
		label_val=leaf.iloc[idx,label_index]
		if label_val not in  label_count.keys():
			label_count[label_val]=0
		label_count[label_val]+=1
	#print label_count
	sort_label=sorted(label_count.iteritems(),key=lambda d:d[1],reverse=True)
	#if len(sort_label)<1 or len(sort_label[0])<1:
	#print leaf
	#print "%d"%len(sort_label)
	#print "%d"%len(sort_label[0])
	max_label=sort_label[0][0]
	max_count=sort_label[0][1]
	prob=float(max_count)/total_num
	#print "major:%r, prob:%f 1-prob:%f" % (max_label,prob,1-prob)
	return (max_label,prob,1-prob)

def construct_tree(DataSet,feature_set):
	label_set=DataSet.iloc[:,label_index].values
	if len(set(label_set))==1:
		#print "only one kind of label,create a leaf"
		return (label_set[0],1,0)
	if len(feature_set)==0:
		#print "all feature done,create a leaf"
		return select_major(DataSet)

	best_feature_idx=select_opt_feature(DataSet,feature_set)
	best_feature=feature_set[best_feature_idx]
	feature_val_list=set(DataSet[best_feature])
	sub_feature_set=cp.deepcopy(feature_set)
	del sub_feature_set[best_feature_idx]
	tree=(best_feature,{})
	for feature_val in feature_val_list:
		#print "create subnode at %r=%r" % (best_feature,feature_val)
		#print sub_feature_set
		sub_DataSet=DataSet[DataSet[best_feature]==feature_val]
		sub_tree=construct_tree(sub_DataSet,sub_feature_set)
		tree[1][feature_val]=sub_tree

	return tree


decision_node=dict(boxstyle="sawtooth",fc="0.8")
leaf_node=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")
node_dist=0.15
layer_dist=0.15
root_pos=(0.5,1)

def plot_node(node,pos,root_flag):
	node_pos=cp.deepcopy(pos)
	parent_pos=node[1]
	if root_flag==True:
		node_pos=root_pos
		parent_pos=root_pos
	node_info=node[0]
	feature_val=node[2]
	#print "Node info"
	#print node_info
	if type(node_info[1]).__name__=='dict':
		#print "here"
		node_text=node_info[0]
		node_type=decision_node
	else:
		#print "there"
		#node_text="%s:%r Prob:%.2f"%(label_name,node_info[0],node_info[1])
		node_text="%r:%.2f"%(node_info[0],node_info[1])
		node_type=leaf_node
	#print "node pos %f %f" %node_pos
	#print "parent ps %f %f" % parent_pos
	plot_tree.ax1.annotate(node_text,xy=parent_pos,xycoords='axes fraction',xytext=node_pos,textcoords='axes fraction',va="center",ha="center",bbox=node_type,arrowprops=arrow_args)
	if root_flag==False:
		x_c=(node_pos[0]+parent_pos[0])/2
		y_c=(node_pos[1]+parent_pos[1])/2
		plot_tree.ax1.text(x_c,y_c,feature_val,va="center",ha="center",rotation=10)

def plot_tree(tree):
	fig=plt.figure(1,facecolor='white')
	fig.clf()
	plt.axis((-5,5,-5,1))
	plot_tree.ax1=plt.subplot(111,frameon=False)
	node_queue=[]
	layer_num=0
	#note,parent pos
	node_queue.append((tree,(0,0),'0'))
	while len(node_queue)>0:
		new_queue=[]
		node_num=len(node_queue)
		right_bound=-1*node_dist*float(node_num-1)/2
		#print "RB:%f" %right_bound
		#plot one level
		node_idx=0
		for node in node_queue:
			if layer_num==0:
				root_flag=True
			else:
				root_flag=False
			node_pos=(root_pos[0]+right_bound+node_idx*node_dist,root_pos[1]-layer_num*layer_dist)
			if root_flag==True:
				node_pos=root_pos
			plot_node(node,node_pos,root_flag)
			node_content=node[0]
			node_type=node[0][1]
			root_flag=False
			#print node_content
			if type(node_type).__name__=='dict':
				sub_dict_tree=node_content[1]
				for feature in sub_dict_tree:
					new_node=(sub_dict_tree[feature],node_pos,feature)
					new_queue.append(new_node)
			node_idx+=1
		layer_num+=1
		node_queue=cp.deepcopy(new_queue)
		#print "queue len:%d" % len(node_queue)

	plt.show()

def select_label(decision_tree,data_vector):
	curr_node=decision_tree
	feature_name=curr_node[0]
	feature_list=curr_node[1]
	#print "name %r" % feature_name
	#print feature_list
	#print data_vector
	while type(feature_list).__name__=='dict':
		#not a leaf
		feature_val=data_vector[feature_name]
		if feature_val not in feature_list.keys():
			print "error @ %r %r" %(feature_name,feature_val)
			print feature_list
			return 0
		curr_node=feature_list[feature_val]
		feature_name=curr_node[0]
		feature_list=curr_node[1]

	return curr_node[0]


