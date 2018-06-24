import tensorflow as tf
import NeuralNetwork

class ImageClassifier(NeuralNetwork):

    optimizer = None
    cost = None
    accuracy = None

    def __init__(self, train_f, train_l, valid_f, valid_l, test_f, test_l,image_shape, n_classes):
        self.super().__init__(train_f, train_l, valid_f, valid_l, test_f, test_l)
        self.image_shape = image_shape
        self.n_classes = n_classes

        self.build_graph()

        return self

    def input_image(self, image_shape):
        """
        Return a Tensor for a batch of image input
        : image_shape: Shape of the images (3 element list)
        : return: Tensor for image input.
        """
        return self.input_batch(shape=(None, image_shape[0], image_shape[1], image_shape[2]), name="x")


    def input_labels(self, n_classes):
        """
        Return a Tensor for a batch of label input
        : n_classes: Number of classes
        : return: Tensor for label input.
        """
        return self.input_batch(shape=(None, n_classes), name="y")


    def keep_prob(self):
        """
        Return a Tensor for keep probability
        : return: Tensor for keep probability.
        """
        return self.input_scalar(name="keep_prob")

    def get_batch(self):
        pass


    def conv_layer(self,x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
        """
        Apply convolution then max pooling to x_tensor
        :param x_tensor: TensorFlow Tensor
        :param conv_num_outputs: Number of outputs for the convolutional layer
        :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
        :param conv_strides: Stride 2-D Tuple for convolution
        :param pool_ksize: kernal size 2-D Tuple for pool
        :param pool_strides: Stride 2-D Tuple for pool
        : return: A tensor that represents convolution and max pooling of x_tensor
        """

        shape = x_tensor.get_shape().as_list()
        filter_depth = shape[3]

        #Initialize weights and biases
        W = self.weights(
            (conv_ksize[0], conv_ksize[1], filter_depth, conv_num_outputs),
            stddev=0.5,
            name='W_conv')

        b = self.biases(conv_num_outputs, name='b_conv')

        conv_layer = self.conv2d(x_tensor,
                                weights=W,
                                biases=b,
                                strides=conv_strides,
                                padding='SAME')


        conv_layer = self.max_pool(conv_layer,
                                    ksize = pool_ksize,
                                    strides = pool_strides,
                                    padding = 'SAME')


        return conv_layer


    def output(self, x_tensor, num_outputs):
        """
        Apply a output layer to x_tensor using weight and bias
        : x_tensor: A 2-D tensor where the first dimension is batch size.
        : num_outputs: The number of output that the new tensor should be.
        : return: A 2-D tensor where the second dimension is num_outputs.
        """
        input_shape = x_tensor.get_shape().as_list()
        W = self.weights((input_shape[1], num_outputs),
                            stddev=0.05,
                            name='W_output')

        b = self.biases(num_outputs, name='b_output')

        output = tf.add( tf.matmul(x_tensor, W), b)

        return output

    def conv_net(self, x, outputs, keep_prob=0.8):
        """
        Create a convolutional neural network model
        : x: Placeholder tensor that holds image data.
        : keep_prob: Placeholder tensor that hold dropout keep probability.
        : return: Tensor that represents logits
        """
        with tf.name_scope('Conv_Layer'):

            conv_ksize = [3, 3]
            conv_strides = [1, 1]

            n_outputs = 64
            pool_ksize = [1, 1]
            pool_strides = [2, 2]

            x = self.conv_layer(x, n_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

            n_outputs = 128
            pool_ksize = [2, 2]
            x = self.conv_layer(x, n_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

            x = self.flatten(x)

        with tf.name_scope('Fully_Conn_Layer'):
            x = self.fully_conn(x, 1000)
            x = self.drop_out(x, keep_prob)

        with tf.name_scope('Output_Layer'):
            x = self.output(x, outputs)

            # Name logits Tensor, so that is can be loaded from disk after training
            logits = self.identity(x, name='logits')

        return logits

    def build_graph(self):
        """
        Build the image classification network graph
        :return:
        """

        #Clear all variables on graph
        self.reset_graph()

        #Build up graph
        x = self.input_image(self.image_shape)
        y = self.input_labels(self.n_classes)
        keep_prob = self.keep_prob()

        # Model
        logits = self.conv_net(x, self.n_classes, keep_prob)

        # Loss and Optimizer
        self.cost = self.get_cost(logits=logits, labels=y, type='softmax')
        self.optimizer = self.optimize(self.cost)

        # Accuracy
        self.accuracy = self.get_accuracy(logits, y)

        #Save graph layout
        self.write_graph()


    def train(self, epochs=1000, batch_size=256, keep_prob=0.8):

        self.super().train( self.get_batch,
                            self.optimizer,
                            self.cost,
                            self.accuracy,
                            epochs=epochs,
                            batch_size = batch_size,
                            keep_prob = keep_prob)