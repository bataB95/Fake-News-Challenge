import tensorflow as tf
import numpy as np
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
data=pickle.load(open("processed_data.p","rb"))
train_set=[]
test_set=[]
size =len(data)
test_size=size//3
#print(data)
for sample in data[0:test_size]:
	if sample[1]!=0:
		test_set.append([list(sample[0]),sample[1]-1])
for sample in data[test_size:size-1]:
	if sample[1]!=0:
		train_set.append([list(sample[0]),sample[1]-1])
print(len(train_set[0][0]))
feature_columns=[tf.contrib.layers.real_valued_column("",dimension=50)]
classifier=tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,hidden_units=[60],n_classes=3,activation_fn=tf.nn.relu)
def get_train_set():
	x=tf.constant([train_set[i][0] for i in range(len(train_set))])
	y=tf.constant([train_set[i][1] for i in range(len(train_set))])
	return x,y
def get_test_set():
	x=tf.constant([test_set[i][0] for i in range(len(test_set))])
	y=tf.constant([test_set[i][1] for i in range(len(test_set))])
	return x,y
classifier.fit(input_fn=get_train_set,steps=1000)
accuracy_score=classifier.evaluate(input_fn=get_test_set,steps=1)["accuracy"]
print("\nTest Accuracy:{0:f}\n".format(accuracy_score))
accuracy_score=classifier.evaluate(input_fn=get_train_set,steps=1)["accuracy"]
print("\nTrain Accuracy:{0:f}\n".format(accuracy_score))