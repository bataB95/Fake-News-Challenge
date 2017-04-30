import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import pickle

class dataWrapper:
    def __init__(self,data):
        self.size=len(data)
        self.x_title=[]
        self.x_body=[]
        self.y=[]
        self.seqlen_title=[]
        self.seqlen_body=[]
        self.current_batch=0
        for sample in data:
            self.x_title.append(sample[0])
            self.x_body.append(sample[1])
            self.y.append(sample[2])
            self.seqlen_title.append(len(sample[0]))
            self.seqlen_body.append(len(sample[1]))
    # return the next batch of the data from the data container
    def next(self,batch_size):
        if self.current_batch+batch_size<self.size:
            self.current_batch+=batch_size
            return self.x_title[self.current_batch:self.current_batch+batch_size],self.x_body[self.current_batch:self.current_batch+batch_size],self.y[self.current_batch:self.current_batch+batch_size],self.seqlen_title[self.current_batch:self.current_batch+batch_size],self.seqlen_body[self.current_batch:self.current_batch+batch_size]
        else:
            self.current_batch=self.current_batch+batch_size-self.size
            batch_x_title=np.concatentate(self.x_title[self.current_batch:],self.x_title[:self.current_batch])
            batch_x_body=np.concatentate(self.x_body[self.current_batch:],self.x_body[:self.current_batch])
            batch_y=np.concatenate(self.y[self.current_batch:],self.y[:self.current_batch])
            batch_seqlen_title=np.concatenate(self.seqlen_title[self.current_batch:],self.seqlen_title[:self.current_batch])
            batch_seqlen_body=np.concatenate(self.seqlen_body[self.current_batch:],self.seqlen_body[:self.current_batch])
            return batch_x_title,batch_x_body,batch_y,batch_seqlen_title,batch_seqlen_body
    def max_seqlen(self):
        return max(self.seqlen_body+self.seqlen_title)
data=pickle.load(open("data.p","rb"))
size=len(data)
trainset=dataWrapper(data[size//3:])
testset=dataWrapper(data[:size//3])

learning_rate=0.001
training_iters=100000
batch_size=128
display_step=10

seq_max_len=max(trainset.max_seqlen(),testset.max_seqlen())
n_input=50
n_hidden=60
n_classes=4

x_title=tf.placeholder("float",[None,seq_max_len,n_input])
x_body=tf.placeholder("float",[None,seq_max_len,n_input])
y=tf.placeholder("float",[None,n_classes])
seqlen_title=tf.placeholder(tf.int32,[None])
seqlen_body=tf.placeholder(tf.int32,[None])
weights={'out':tf.Variable(tf.random_normal([n_hidden,n_classes]))}
biases={'out':tf.Variable(tf.random_normal([n_classes]))}

def dynamicRNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases):
    x_title=tf.unstack(x_title,seq_max_len,1)
    x_body=tf.unstack(x_body,seq_max_len,1)


    with tf.variable_scope('scope1'):
        lstm_cell=rnn.BasicLSTMCell(n_hidden)
    outputs_title,states_title=rnn.static_rnn(cell=lstm_cell,inputs=x_title,sequence_length=seqlen_title,dtype=tf.float32)
    outputs_body,states_body=rnn.static_rnn(cell=lstm_cell,input=x_body,sequence_length=seqlen_body,dtype=tf.float32)

    return tf.matmul(tf.mul(outputs_title[-1],outputs_body[-1]),weight['out'])+biases['out']

pred=dynamicRNN(x_title,x_body,seqlen_title,seqlen_body,weights,biases)
cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step=1

    while step*batch_size<training_iters:
        print(step)
        batch_x_title,batch_x_body,batch_y,batch_seqlen_title,batch_seqlen_body=trainset.next(batch_size)
        sess.run(optimizer,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body})
        if step%display_step==0:
            acc=sess.run(accuracy,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body})
            loss=sess.run(cost,feed_dict={x_title:batch_x_title,x_body:batch_x_body,y:batch_y,seqlen_title:batch_seqlen_title,seqlen_body:batch_seqlen_body})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step+=1
    print("Optimization Finished!")

test_x_title=testset.x_title
test_x_body=testset.x_body
test_y=testset.y
test_seqlen_title=testset.seqlen_title
test_seqlen_body=testset.seqlen_body

print("Test Accuracy:",sess.run(accuracy,feed_dict={x_title:test_x_title,x_body:test_x_body,y:test_y,seqlen_title:test_seqlen_title,seqlen_body:test_seqlen_body}))
   