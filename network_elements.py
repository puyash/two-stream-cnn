import tensorflow as tf
import numpy as np
import os

class convLayer(object):
    '''
    Generates a convolutional layer with a 4d input and 4d output volume
    shape=[nrow, ncol, chanels, nfilters]
    strides=[1, nrow, ncol, 1]
    '''

    def __init__(self, shape, strides, name, padding="SAME"):
        self.shape=shape
        self.strides=strides
        self.name=name
        self.padding=padding

        # set weight var
        winit=tf.truncated_normal(shape, stddev=0.1)
        self.weight=tf.Variable(winit, name="w_{}".format(name))
        # set bias var
        binit=tf.constant(0.1, shape=[shape[-1]])
        self.bias=tf.Variable(binit, name="b_{}".format(name))

    def layer(self, x_in):
        weighted=tf.nn.conv2d(x_in, self.weight, strides=self.strides,
                                padding=self.padding)
        weighted_bias=tf.add(weighted, self.bias)

        return weighted_bias

    def layer_relu(self, x_in):
        # layer activations with relu and dropout
        activation=tf.nn.relu(self.layer(x_in))
        return activation


class fullLayer(object):
    '''
    Generates a traditional fully conected layer with a 2d input and 2d output volume
    '''

    def __init__(self, shape, name):
        self.shape=shape
        self.name=name
        # set weight var
        winit=tf.truncated_normal(shape, stddev=0.1)
        self.weight=tf.Variable(winit, name='w_{}'.format(name))
        # set bias var
        binit=tf.constant(0.1, shape=[shape[-1]])
        self.bias=tf.Variable(binit, name='b_{}'.format(name))

    def layer(self, x_in, flat=True):
        if flat:
            x_in=tf.reshape(x_in, [-1, self.shape[0]])

        weighted=tf.matmul(x_in, self.weight)
        weighted_bias=tf.add(weighted, self.bias)

        return weighted_bias

    def layer_relu(self, x_in, keep_prob):
        # layer activations with relu and dropout
        activation=tf.nn.dropout(tf.sigmoid(self.layer(x_in)), keep_prob)
        return activation

    def layer_sigmoid(self, x_in, keep_prob):
        # layer activations with sigmoid and dropout
        activation=tf.nn.dropout(tf.sigmoid(self.layer(x_in)), keep_prob)
        return activation


class maxPool(object):
    def __init__(self, ksize, strides, padding='SAME'):
        self.ksize=ksize
        self.strides=strides
        self.padding=padding

    def pool(self, x, keep_prob):
        pooled=tf.nn.dropout(tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides,
                                padding=self.padding), keep_prob)
        self.outdim=pooled.get_shape().as_list()
        self.numout=np.prod(self.outdim[-3:])
        return pooled


class Dataset():
    
    def __init__(self, x1, x2, y, testsize=0.2,  shuffle=True):
        
        leny=len(y)
        
        if shuffle == True:
            si=np.random.permutation(np.arange(leny))
            x1=x1[si]
            x2=x2[si]
            y =y [si]
        
        if type(testsize) == int:
            testindex=testsize
        else: 
            testindex=int(testsize*leny)     
                
            self.x1_train=x1[testindex:]; self.x1_test=x1[:testindex] 
            self.x2_train=x2[testindex:]; self.x2_test=x2[:testindex]
            self.y_train =y [testindex:]; self.y_test =y [:testindex]
	    
	    print('Train size: {}, test size {}'.format(len(self.y_train),len(self.y_test)))
    
 
class modelSaver(object):
    def __init__(self, mdir, max_save):
        self.mdir=mdir
        if not os.path.exists(mdir):
            os.makedirs(mdir)
        self.saver=tf.train.Saver(max_to_keep=max_save)
        
    def save(self, sess, name, current_step, print_info):
        savestring=self.mdir + name 
        self.saver.save(sess, savestring, global_step=current_step)
        if print_info:
            print("Checkpoint saved at: {}".format(self.mdir))   




def network(x1, x2, keep_prob, params):
    
    x1=tf.expand_dims(x1, 3)
    x2=tf.expand_dims(x2, 3)
    
    # convolutional ---------------------------------             
    # -----------------------------------------------
    
    with tf.name_scope('convlayer_1'):
        # convolutional layer ----------------------
        # initialize convolutional layer
        convlayer1=convLayer(shape=params['conv']['shape'],
                               strides=params['conv']['strides'], name="conv")
        h_conv1=convlayer1.layer_relu(x1)
    
        
    with tf.name_scope('poollayer_1'):
        # pooling layer ----------------------------
        poollayer1=maxPool(ksize=params['pool']['ksize'], strides=params['pool']['strides'])
        h_pool1=poollayer1.pool(h_conv1, keep_prob)
    
    # -----------------------------------------------------------------------------------
    
    with tf.name_scope('convlayer_2'):
        # convolutional layer ----------------------
        # initialize convolutional layer
        convlayer2=convLayer(shape=params['conv']['shape'],
                              strides=params['conv']['strides'], name="conv")
        h_conv2=convlayer2.layer_relu(x2)
        
    
    with tf.name_scope("poollayer_2"):
        # pooling layer ----------------------------
        poollayer2=maxPool(ksize=params['pool']['ksize'], strides=params['pool']['strides'])
        h_pool2=poollayer2.pool(h_conv2, keep_prob)
        
    
    # fully connected -------------------------------             
    # -----------------------------------------------
    
    # concatenate output of the two parallel conv layers      
    full_in=tf.concat(3, [h_pool1, h_pool2])
    num_full_in=poollayer1.numout + poollayer2.numout
    
    
    with tf.name_scope('fullayer_combined'):
        # fully connected layer---------------------
        # initialize hidden fully connected layer
        fullayer=fullLayer(shape=[num_full_in, params['full']], name='full')
        h_full=fullayer.layer_relu(full_in, keep_prob)
        
    
    with tf.name_scope('outlayer'):
        # output layer -----------------------------
        # initialize output layer
        outlayer=fullLayer(shape=params['vout'], name='out')
        v_out=outlayer.layer(h_full)
        # ------------------------------------------

        l2_loss=tf.nn.l2_loss(convlayer1.weight) + tf.nn.l2_loss(convlayer2.weight) + \
        tf.nn.l2_loss(fullayer.weight) + tf.nn.l2_loss(outlayer.weight)

    return v_out, l2_loss





def optimization_ops(y, Y, l2_loss, reg_param=0.0, learningrate=0.001):
    '''Generates optimization related operations'''
    with tf.name_scope('cost'):
        cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, Y)) + reg_param * l2_loss

    with tf.name_scope('optimization'):
        optimize=tf.train.AdamOptimizer(learningrate)
        grads_vars=list(zip(tf.gradients(cost, tf.trainable_variables()), tf.trainable_variables()))
        train_step=optimize.apply_gradients(grads_and_vars=grads_vars)

 
    return(cost, grads_vars, train_step)



def two_stream_batches(X1, X2, Y, batch_size, num_epochs, shuffle=True):


    data_size=len(Y)

    num_batches_per_epoch=int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
    # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            X1=X1[shuffle_indices]
            X2=X2[shuffle_indices]
            Y=Y[shuffle_indices]

        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num * batch_size
            end_index=min((batch_num + 1) * batch_size, data_size)

            yield X1[start_index:end_index], X2[start_index:end_index], Y[start_index:end_index] 





