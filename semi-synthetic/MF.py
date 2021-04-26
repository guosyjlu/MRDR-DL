"""
Standard Matrix Factorization implemented with TensorFlow.
"""
import tensorflow as tf


class MF(object):
    def __init__(self, user_num, item_num, embedding_size, l2_reg_lambda):
        self.user_id = tf.placeholder(tf.int32, [None])
        self.item_id = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.float32, [None])

        initializer = tf.random_normal_initializer()
        self.user_embedding = tf.Variable(initializer([user_num, embedding_size]))
        self.item_embedding = tf.Variable(initializer([item_num, embedding_size]))
        self.user_bias = tf.Variable(initializer([user_num]))
        self.item_bias = tf.Variable(initializer([item_num]))
        self.global_bias = tf.Variable(initializer([1]))

        self.user_feature = tf.nn.embedding_lookup(self.user_embedding, self.user_id)
        self.item_feature = tf.nn.embedding_lookup(self.item_embedding, self.item_id)
        self.b_u = tf.nn.embedding_lookup(self.user_bias, self.user_id)
        self.b_i = tf.nn.embedding_lookup(self.item_bias, self.item_id)
        self.prediction = tf.reduce_sum(tf.multiply(self.user_feature, self.item_feature), 1)
        self.prediction += self.b_u + self.b_i + self.global_bias

        self.mse = tf.reduce_mean(tf.square(self.prediction-self.y))
        self.l2_regularization = tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding)
        self.l2_regularization += tf.nn.l2_loss(self.user_bias) + tf.nn.l2_loss(self.item_bias)
        self.l2_regularization += tf.nn.l2_loss(self.global_bias)
        self.loss = self.mse + l2_reg_lambda * self.l2_regularization
