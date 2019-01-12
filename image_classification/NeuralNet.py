import tensorflow as tf
import time
import numpy as np


class NeuralNet:
    log_model_path = './tmp/logs/2/train'
    save_model_path = './tmp/weights'

    def __init__(self, features, labels, valid_features, valid_labels, n_layers=1, n_hidden=10, lr=0.1, epochs=100,
                 batch_size=128, show_every=10):
        self.features = features
        self.outputs = labels

        self.valid_features = valid_features
        self.valid_labels = valid_labels

        self.n_examples, self.n_features = features.shape
        _, self.n_outputs = labels.shape

        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.show_every = show_every

    def inputs(self):
        """
        Return a Tensor for feature inputs
        : return: Tensor for input features.
        """
        return tf.placeholder(tf.float32, shape=[None, self.n_features], name="x")

    def labels(self):
        """
        Return a Tensor for input labels
        :return: Tensor for labels
        """
        return tf.placeholder(tf.float32, shape=[None, self.n_outputs], name="y")

    def learning_rate(self):
        """
        Return a Tensor for setting the learning rate
        :return: Variable tensor for learning rate
        """
        return tf.placeholder(tf.float32, name="lr")

    def get_batch(self, batch_size=128):
        batch = np.random.choice(range(self.n_examples), size=batch_size)
        x, y = self.features[batch], self.outputs[batch]
        return x, y

    def add_layer(self, inputs, n_outputs, activation=''):
        """
        Adds a fully connected layer to the graph
        :param inputs: Tensor that are inputs to layer
        :param n_outputs: Number of of outputs
        :param activation: Activation function to use. Default includes no activation.
        :return: Tensor representing fully connected layer
        """

        input_shape = inputs.get_shape().as_list()

        W = tf.Variable(tf.random_normal((input_shape[1], n_outputs), stddev=0.05), name='W')
        b = tf.Variable(tf.zeros(n_outputs), name='b')

        fully_conn = tf.add(tf.matmul(inputs, W), b)

        # Add activation functions if specified
        if activation == 'RELU':
            fully_conn = tf.nn.relu(fully_conn, name='RELU')
        elif activation == 'sigmoid':
            fully_conn = tf.sigmoid(fully_conn, name='Sigmoid')

        tf.summary.histogram('W', W)
        tf.summary.histogram('b', b)

        return fully_conn
        # return tf.contrib.layers.fully_connected(inputs, n_outputs, activation_fn=tf.nn.relu)

    def build_graph(self):
        tf.reset_default_graph()

        # Data
        with tf.name_scope('Inputs'):
            self.x = self.inputs()
            self.y = self.labels()
            self.alpha = self.learning_rate()

        # Network
        with tf.name_scope('Network'):

            # Hidden Layers
            for i in range(self.n_layers):
                if i == 0:
                    self.hidden = self.x
                self.hidden = self.add_layer(self.hidden, self.n_hidden, activation="RELU")

            # Output Layer
            self.logits = self.add_layer(self.hidden, self.n_outputs, activation="")
            tf.summary.histogram('logits', self.logits)

        # Cost & Optimizer
        with tf.name_scope('Cost'):
            self.cost = tf.reduce_mean(tf.square(self.logits - self.y), name='cost')
            tf.summary.scalar('cost', self.cost)

        with tf.name_scope('Train'):
            self.optimizer = tf.train.AdamOptimizer(self.alpha).minimize(self.cost)
            # self.optimizer = tf.train.GradientDescentOptimizer(self.alpha).minimize(self.cost)

        # Collect all plots
        self.summary = tf.summary.merge_all()

    def train(self):

        self.build_graph()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            train_writer = tf.summary.FileWriter(self.log_model_path, sess.graph)

            for e in range(self.epochs):

                start = time.time()

                # Get Batch
                x, y = self.get_batch()

                # Train Model
                summary, cost, _ = sess.run([self.summary, self.cost, self.optimizer],
                                            feed_dict={self.x: x,
                                                       self.y: y,
                                                       self.alpha: self.lr}
                                            )

                # Validation Cost
                valid_cost = sess.run(self.cost, feed_dict={self.x: self.valid_features,
                                                            self.y: self.valid_labels,
                                                            self.alpha: self.lr})

                end = time.time()

                if (e % self.show_every) == 0:
                    print('Epoch {}/{} '.format(e + 1, self.epochs),
                          'Training loss: {:.4f}'.format(cost),
                          'Validation loss: {:.4f}'.format(valid_cost),
                          '{:.4f} sec/epoch'.format((end - start)))

                # Write Log Outputs
                train_writer.add_summary(summary, e)

            # Save Model
            saver = tf.train.Saver()
            saver.save(sess, self.save_model_path)

    def predict(self, features):

        # Build graph to load back into
        self.build_graph()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Reload graph variables
            saver.restore(sess, self.save_model_path)

            # Run predictions
            logits = sess.run(self.logits, feed_dict={self.x: features})
            return logits
