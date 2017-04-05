# internal modules
from network_elements import *
from two_stream import *
# external modules
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
import gensim
from datetime import datetime



np.random.seed(66)

# ---------------------------------------------------
# network parameters --------------------------------

REG_PARAM=0.0
KEEP_PROB=0.5
LEARNING_RATE=0.001

# fully connected
NUM_HIDDEN_UNITS=512
NUM_CLASSES=2
# filters and pooling
NUM_FILTERS=32
# conv filters
FILTER_WIDTH=5
FILTER_HEIGHT=5
# pooling
POOL_WIDTH=2
POOL_HEIGHT=2 

NETWORK_PARAMS={
    "conv" : {"shape" : [FILTER_HEIGHT, FILTER_WIDTH, 1, NUM_FILTERS], "strides" : [1,1,1,1]},
    "pool" : {"ksize" : [1,POOL_HEIGHT,POOL_WIDTH,1], "strides" : [1,POOL_HEIGHT,POOL_WIDTH,1]}, 
    "full" : NUM_HIDDEN_UNITS,
    "vout" : [NUM_HIDDEN_UNITS, 2]
}

# ---------------------------------------------------
# get mnist data ------------------------------------

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('/tmp/data/', one_hot=False)

X1=mnist.train.images
X1=np.reshape(X1, [-1, 28, 28])
Ytemp=mnist.train.labels

# ---------------------------------------------------
# generate second data array ------------------------

# Generate array X2 where each element is either a randomly
# selected non-matching digit or randomly selected matching digit
X2, Y=generate_complement(X1, Ytemp, 0.5)    

# final data object
data=Dataset(X1, X2, Y, testsize=0.2,  shuffle=True)


# ---------------------------------------------------
# build network -------------------------------------

tf.reset_default_graph() 


# input train vars
with tf.name_scope('input'):
    x1=tf.placeholder(tf.float32, [None, 28, 28], name="X1") # add dim for multiple channels
    x2=tf.placeholder(tf.float32, [None, 28, 28], name="X2")
    y =tf.placeholder(tf.float32, [None, 2], name="Y")

    keep_prob=tf.placeholder(tf.float32)



with tf.name_scope('network'):
    yh, l2_loss=network(x1, x2, keep_prob, NETWORK_PARAMS)

with tf.name_scope('opt-ops'):
    cost, grads_vars, train_step=optimization_ops(yh, y, l2_loss, REG_PARAM, LEARNING_RATE)

with tf.name_scope('metrics'):    
    
    # accuracy        
    correct_prediction=tf.equal(tf.argmax(yh, 1), tf.argmax(y, 1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # auc
    pr=tf.cast(tf.argmax(tf.nn.softmax(yh), 1),tf.float32)
    yl=tf.cast(tf.argmax(y, 1),tf.float32)
    
    auc, update_op=tf.contrib.metrics.streaming_auc(pr,yl)
    
# ---------------------------------------------------
# set tensorboard ops  ------------------------------

init_all=tf.initialize_all_variables()
init_loc=tf.initialize_local_variables()

# cost summary
tf.scalar_summary('cost', cost)
# acc summary
tf.scalar_summary('accuracy', accuracy)
# weights and biases summary
for var in tf.trainable_variables():
    tf.histogram_summary(var.name, var)

# gradient summary
for grad, var in grads_vars:
    tf.histogram_summary(var.name + '/gradient', grad)

merged_summaries=tf.merge_all_summaries()

# ---------------------------------------------------
# run operations  -----------------------------------



step=1
test_interval=10
save_interval=1000
acc_test=[]
acc_train=[]


with tf.Session() as sess:
    # initialize variables, sumaries and checkpoint saver
    sess.run([init_all, init_loc])
    summary_writer=tf.train.SummaryWriter('./summaries', graph=tf.get_default_graph())
    modelsaver=modelSaver('./models/', 100)

    
    print("----------------------- Training starts -------------------------")

    batches=two_stream_batches(data.x1_train, data.x2_train, data.y_train, 100, 20)
    
    for batch in batches:
        x1_batch, x2_batch, y_batch=batch
        
        # perform training step
        _, summary=sess.run([train_step, merged_summaries], 
                              feed_dict={x1: x1_batch, x2: x2_batch, y: y_batch, keep_prob: KEEP_PROB})
        # write summaries
        summary_writer.add_summary(summary, step)

        # calculate validation metrics
        if step % test_interval == 0 or step == 1:
            
            te_acc, te_cost=sess.run([accuracy, cost], 
                                        feed_dict={x1: data.x1_test, x2: data.x2_test, y: data.y_test, keep_prob: 1})
            tr_acc, tr_cost=sess.run([accuracy, cost], 
                                        feed_dict={x1: x1_batch, x2: x2_batch, y: y_batch, keep_prob: 1})
            
            acc_test.append(te_acc)
            acc_train.append(tr_acc)
            
            time=datetime.now().replace(microsecond=0)
            
            print("step: {}, {}, train_acc: {}, test_acc: {}, train_loss: {}, test_loss: {}".format(
                step, time,  round(tr_acc, 3), round(te_acc, 3), round(tr_cost, 3), round(te_cost, 3)))
        
        # save checkpoints
        if step % save_interval == 0:
            modelsaver.save(sess, 'mnist_example', current_step=step, print_info=False)   
            
        step += 1
        
    # final predictions run on separate set    
    predictions=sess.run(tf.nn.softmax(yh), 
                           feed_dict={x1: data.x1_test, x2: data.x2_test, y: data.y_test, keep_prob: 1})        
   
  
