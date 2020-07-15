import tensorflow as tf
import numpy as np
import time
from augment_image_names import Generator
from data_loader import load_batch
import matplotlib.pyplot as plt

"""

https://github.com/richzhang/colorization/blob/master/colorization/models/colorization_deploy_v2.prototxt

This object is a convolutional neural network using tensorflow and a generator
with data autmentation. The object initializes with scalars height, width, 
channels, output size, learning rate, and l2 regularization. It also has a list 
of ints as the units for the convoluational layers, fully connected layers, 
kernel size, pool and convolutional stride lengths. There is also a list of 
floats for the keep probability values for each fully connected layer. Finally, 
there is a variable for the fully connected activation function and the type of 
output (classification vs regression). All of the lists pertaining to the 
convolutional (and pooling) layers must be of the same length. The same holds 
true for the fully connected layers. Finally, there is the activation function, 
which stays consistent across all fully connected layers (except for the output 
layer). The network takes batches of image data, puts it through the 
convolutional layers (each followed by a pooling layer) with parameters denoted 
by the conv_featmap, kernel_size, conv_strides, pool_size, and pool_strides 
lists. This output is then funneled to the fully connected layers with parameter
lists fc_layer_size and train_keep_prob. The output of these layers is 
determined by the output size int. The training function of the object takes 
batches of any size but must be an NxM matrix input. Using the train function, 
users can train on presplit data and can alter the parameters of batch size, 
number of epochs, translation, flips, rotations, and added noise. This trained 
network is automatically saved under the given name and can be loaded for 
further training. There is also a predict function that loads an input pre 
trained model or loads the most recent checkpoint for predictions.
"""

class cnn(object):
    def __init__(self,height,width,channels,conv_featmap=[32,64],deconv_featmap=[64,32,3],
                 kernel_size=[5,5],dekernel_size=[5,5,5],conv_strides=[1,1],deconv_strides=[1,1,1],
                 pool_size=[2,2],pool_strides=[2,2],upsample_size=[2,2],learning_rate=0.01,
                 lambda_l2_reg=.01,activation=tf.nn.relu):
        # Ensures that the hidden layers have corresponding keep probs
        assert len(conv_featmap) == len(kernel_size) and len(conv_featmap) == len(pool_size)


        # Sets variables for later use
        self.height = height
        self.width = width
        self.channels = channels
        self.conv_featmap = conv_featmap
        self.deconv_featmap = deconv_featmap
        self.kernel_size = kernel_size
        self.dekernel_size = dekernel_size
        self.conv_strides = conv_strides
        self.deconv_strides = deconv_strides
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.upsample_size = upsample_size
        self.learning_rate = learning_rate
        self.lambda_l2_reg = lambda_l2_reg
        self.activation = activation

        # Creates NN using private functions
        self.__input_layer()
        self.__conv_layers()
        self.__deconv_layers()
        self.__output_layer()
        self.__loss()
        self.__optimizer()
        self.saver = tf.train.Saver()

    # Creates the input layer using placeholders (assumes batches x 3 dim input)
    def __input_layer(self):
        self.inputs = tf.placeholder(tf.float32,shape=(None,self.height,self.width,self.channels))
        self.targets = tf.placeholder(tf.int64,shape=(None,self.height,self.width,3))
        self.output_tf = tf.placeholder(tf.bool,shape(None,self.height,self.width))

    # Creates convolutional layers (each conv layer followed by a pooling layer)
    def __conv_layers(self):
        self.conv = tf.layers.conv2d(self.inputs, filters=self.conv_featmap[0],
                                     kernel_size=self.kernel_size[0],strides=self.conv_strides[0],
                                     padding="SAME",activation=self.activation)
        self.pool = tf.nn.max_pool(self.conv, ksize=[1,self.pool_size[0],self.pool_size[0],1],
                                   strides=[1,self.pool_strides[0],self.pool_strides[0],1],padding="VALID")

        for i in range(1,len(self.conv_featmap)):
            self.conv = tf.layers.conv2d(self.pool, filters=self.conv_featmap[i],kernel_size=self.kernel_size[i],
                                         strides=self.conv_strides[i],padding="SAME",activation=self.activation)
            self.pool = tf.nn.max_pool(self.conv, ksize=[1,self.pool_size[i],self.pool_size[i],1],
                                       strides=[1,self.pool_strides[i],self.pool_strides[i],1],padding="VALID")

    def __deconv_layers(self):
      for i in range(len(self.deconv_featmap)-1):
        self.pool = tf.layers.conv2d(self.deconv, filters=self.deconv_featmap[i],kernel_size=self.dekernel_size[i],
                                              strides=self.deconv_strides[i],padding="SAME",activation=self.activation)
        new_h = tf.shape(self.pool)[1]*self.upsample_size[i]
        new_w = tf.shape(self.pool)[2]*self.upsample_size[i]
        self.deconv = tf.image.resize(self.pool, size=[new_h,new_w])
        
      self.deconv = tf.layers.conv2d(self.deconv, filters=self.deconv_featmap[-1],kernel_size=self.dekernel_size[-1],
                                     strides=self.deconv_strides[-1],padding="SAME",activation=self.activation)

    # Takes output of FC layer and creates an output of output_size
    # (This does not have dropout because it is the output layer)
    def __output_layer(self):
        self.output = tf.multiply(self.output_tf,self.deconv)

    # Defines loss based on if the output is a regression or classification. If 
    # classification, use softmax, if regression, use mean squared error
    def __loss(self):
        self.loss = tf.losses.mean_squared_error(self.targets,self.output)

    # Minimizes the loss using an Adam optimizer
    def __optimizer(self):
        self.optimizer = tf.train.AdamOptimizer(
                         learning_rate=self.learning_rate).minimize(self.loss)

    # Trains the network based on the train inputs given and uses the test set
    # to test accuracy on a non training set
    def train(self,train_names,test_names,base_dir,epochs=20,
              batch_size=64,test_batch_size=64,translate=[0,0],
              flip=[0,0],noise=0,model_name=None,pre_trained_model=None):

        # Create the generator to output batches of data with given transforms
        gen = Generator(train_names,translate=translate,flip=flip,noise=noise)
        next_batch = gen.gen_batch(batch_size)
        
        test_gen = Generator(test_names)
        test_batch = test_gen.gen_batch(test_batch_size)

        # Set number of iterations (SIZE CAN BE CHANGED BECAUSE OF GENERATOR)
        aug_size = gen.aug_size()
        iters = int(aug_size / batch_size)
        print('number of batches for training: {}'.format(iters))
        
        # Set base levels and model name
        iter_tot = 0
        best_acc = 0
        self.losses = []
        if model_name == None:
            cur_model_name = 'basic_model'

        # Start session, initialize variables, and load pretrained model if any
        self.session = tf.Session()
        with self.session as sess:
            merge = tf.summary.merge_all()
            writer = tf.summary.FileWriter("log/{}".format('model'),
                                           self.session.graph)
            sess.run(tf.global_variables_initializer())
            if pre_trained_model != None:
                try:
                    print("Loading model from: {}".format(pre_trained_model))
                    self.saver.restore(sess,'model/{}'.format(pre_trained_model))
                except Exception:
                    raise ValueError("Failed Loading Model")

            # Set up loops for epochs and iterations per epochs
            for epoch in range(epochs):
                print("epoch {}".format(epoch + 1))

                for itr in range(iters):
                    merge = tf.summary.merge_all()
                    iter_tot += 1

                    # Create feed values using the generator
                    feed_names = next(next_batch)
                    feed_image, feed_accels, feed_tf = load_batch(feed_names,base_dir)
                    feed = {self.inputs: feed_image, self.targets: feed_accels, self.output_tf: feed_tf}

                    # Feed values to optimizer and output loss (for printing)
                    _, cur_loss = sess.run([self.optimizer,self.loss],
                                           feed_dict=feed)
                    self.losses.append(cur_loss)

                    # After 100 iterations, check if test accuracy has increased
                    if iter_tot % 100 == 0:
                        feed_test = next(test_batch)
                        test_images, test_accels, test_tf = load_batch(feed_test,base_dir)
                        pred = sess.run([self.pred],feed_dict={self.inputs:
                                        test_images, self.targets: test_accels, 
                                        self.output_tf: test_tf})
                        mse = np.mean((pred-test_accels)**2)
                        if mse < best_mse:
                            print('Best validation accuracy! iteration:'
                                  '{} mse: {}%'.format(iter_tot, mse))
                            best_mse = mse
                            self.saver.save(sess,'model/{}'.format(
                                            cur_model_name))

        print("Traning ends. The best valid accuracy is {}." \
               " Model named {}.".format(best_mse, cur_model_name))

    # Plot training losses from most recent session
    def plot(self):
        plt.plot(self.losses)

    # Predicts class of input based on pre trained model
    def predict(self,x,pre_trained_model=None):
        assert(x.shape[1:] == self.inputs.get_shape()[1:])

        self.session = tf.Session()
        with self.session as sess:
            if pre_trained_model != None:
                try:
                    print("Loading model from: {}".format(pre_trained_model))
                    self.saver.restore(sess,'model/{}'.format(pre_trained_model))
                except Exception:
                    raise ValueError("Failed Loading Model")
            else:
                self.saver.restore(sess,tf.train.latest_checkpoint('model/'))
            pred = sess.run([self.pred],feed_dict={self.inputs: x})
            return pred

