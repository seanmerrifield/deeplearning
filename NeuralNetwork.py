import tensorflow as tf
import numpy as np

class NeuralNetwork():

    def __init__(self):


    def input_batch(self, shape, name, t = "float")
        """
        Return an in placeholder Tensor 
        : shape: Shape of the images
        : name:  Name of the Tensor
        : t:     Type of input ("float" or "int") (Optional)
        : return: A placeholder tensor
        """
        if t is "int":
            t = tf.int32

        
        return tf.placeholder(t, shape=shape, name=name)


    def input_scalar(self, name, t="float"):
         """
        Return an in placeholder scalar Tensor 
        : t:        Type of input ("float" or "int") (Optional)
        : return:   A placeholder scalar tensor
        """
        if t is "int":
            t = tf.int32 

        return tf.placeholder(t, name=name)

    def weights(self, shape, n_outputs, name, std_dev=0.5, ):
          """
            Return weights variable tensor initialized with truncated normal distribution
            : shape:       Shape of variable tensor
            : name:        Name of tensor
            : std_dev:     Standard dev (Optional)
            : return:      A variable tensor with initialized weights
        """   
        W = tf.Variable(
                        tf.truncated_normal(
                        shape=shape,
                        stddev=std_dev), 
                        name=name)

        tf.summary.histogram(name, W)
        return W

    def biases(self, n_outputs, name):
        """
            Return biases variable tensor initialized with zeroes
            : n_outputs:   Number of outputs
            : name:        Name 
            : return:      A variable tensor of biases
        """   
        b = tf.Variable(tf.zeros(n_outputs), name=name)
        tf.summary.histogram(name, b)
        return b
    
    def drop_out(self, tensor, keep_prob=0.8):
        """
            Return biases variable tensor initialized with zeroes
            : tensor:      Tensor to apply drop out to
            : keep_prob:   Keep probability (float) (optional)
            : return:      Tensor with drop out
        """ 
        return tf.nn.dropout(tensor, keep_prob)

    def add_name(self, tensor, name):
        """
            Return tensor with name applied
            : tensor:      Tensor to apply drop out to
            : name:        Name to apply
            : return:      Tensor with a name applied
        """ 
        return tf.identity(tensor, name=name)

    def conv2d(self, inputs, weights, biases, strides, padding='SAME'):
        """
            Return biases variable tensor initialized with zeroes
            : inputs:      Tensor of inputs
            : weights:     Weights variable tensor
            : biases:      Biases variable tensor
            : strides:     Strides in each direction (2 element array)
            : padding:     Padding type (optional)
            : return:      A variable tensor of biases
        """  
        conv_layer = tf.nn.conv2d(inputs, 
                              weights, 
                              strides = [1, strides[0], strides[1], 1], 
                              )
    
        conv_layer = tf.nn.bias_add(conv_layer, biases)
                    
        conv_layer = tf.nn.relu(conv_layer)

        return conv_layer

    def max_pool(self, k_size, stries):
         """
            Return biases variable tensor initialized with zeroes
            : n_outputs:   Number of outputs
            : strides:     Strides in each direction (2 element array)
            : return:      A variable tensor of biases
        """         

        conv_layer = tf.nn.max_pool(conv_layer, 
                                    ksize = [1, pool_ksize[0], pool_ksize[1], 1], 
                                    strides = [1, pool_strides[0], pool_strides[1], 1],
                                    padding = 'SAME')

        return conv_layer


    def fully_conn(self, x_tensor, num_outputs):
        """
        Apply a fully connected layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        
        input_shape = x_tensor.get_shape().as_list()
        
        W = self.weights((input_shape[1], num_outputs), 
                        stddev=0.05), 
                        name='W_full')

        b = self.biases(num_outputs, name = 'b_full')
         
        fully_conn = tf.add( tf.matmul(x_tensor, W), b)
        fully_conn = tf.nn.relu(fully_conn, name = 'fully_conn')

        return fully_conn

    def flatten(x_tensor):
        """
        Flatten x_tensor to (Batch Size, Flattened Image Size)
        : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
        : return: A tensor of size (Batch Size, Flattened Image Size).
        """

        shape = x_tensor.get_shape().as_list()
        dim = np.prod(shape[1:]) 
        
        flattened = tf.reshape(x_tensor, [-1, dim], name = 'flatten')
    
        return flattened

    def get_cost(self, logits, y, type):
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y)
            )

        return cost

    def optimize(self, cost):
        return tf.train.AdamOptimizer().minimize(cost)

    def get_accuracy(self, logits, y):
        # Accuracy
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
        return accuracy

    def train(self, batch_function, optimizer, cost_func, accuracy_func, epochs=1000, batch_size=256, keep_prob=0.8):

        with tf.Session() as sess:
            # Initializing the variables
            sess.run(tf.global_variables_initializer())

            # Training cycle
            for epoch in range(epochs):
                batch_i = 1
                for batch_features, batch_labels in batch_function(batch_i, batch_size):

                        session.run(optimizer, 
                                    feed_dict = {x: batch_features, 
                                                 y: batch_labels, 
                                                 keep_prob: keep_prob}
                                                )

                        loss = session.run(cost, feed_dict = {x: feature_batch, y: label_batch, keep_prob: 1.0})
                        valid_loss = session.run(cost, feed_dict = {x: valid_features, y:valid_labels, keep_prob: 1.0})
                     
                        train_accuracy = session.run(accuracy, feed_dict = {x: feature_batch, y:label_batch, keep_prob: 1.0})
                        valid_accuracy = session.run(accuracy, feed_dict = {x: valid_features, y: valid_labels, keep_prob: 1.0})
                                    

                print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
                self.print_stats(sess, batch_features, batch_labels, cost, accuracy)


    def print_stats(self,loss,train_accuracy,valid_loss,valid_accuracy):

            print('Training Loss: {:.2f}...'.format(loss),
                'Training Accuracy: {:.2f}%...'.format(train_accuracy*100),
                'Validation Loss: {:.2f}...'.format(valid_loss),
                    'Validation Accuracy: {:.2f}%...'.format(valid_accuracy*100))



    def reset_graph(self):
        return  tf.reset_default_graph()


    def write_graph(self, path):
        """
        Write graph out for viewing in tensorboard
        : path: File path for writing tensorboard graph to
        :return: Boolean
        """
        if not path:
            path = './logs/1'

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            file_writer = tf.summary.FileWriter(path, sess.graph)

        return True