"""
The base model: factorization machine.
Because we only take the user-id and item-id as inputs,
FM can also be viewed as a kind of Logistic Matrix Factorization (LMF).
"""
import tensorflow as tf


class logisticMF(object):
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
        self.ctr = tf.squeeze(tf.sigmoid(tf.expand_dims(self.prediction, 1)))

        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)
        self.l2_regularization = tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding)
        self.l2_regularization += tf.nn.l2_loss(self.user_bias) + tf.nn.l2_loss(self.item_bias)
        self.l2_regularization += tf.nn.l2_loss(self.global_bias)
        self.loss = self.cross_entropy + l2_reg_lambda * self.l2_regularization


class LMF(object):
    def __init__(self, user_num, item_num, embedding_size, l2_reg_lambda):
        self.user_id = tf.placeholder(tf.int32, [None])
        self.item_id = tf.placeholder(tf.int32, [None])
        self.p = tf.placeholder(tf.float32, [None])
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
        self.cvr = tf.squeeze(tf.sigmoid(tf.expand_dims(self.prediction, 1)))

        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        self.cross_entropy = tf.div(self.cross_entropy, self.p)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)
        self.l2_regularization = tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding)
        self.l2_regularization += tf.nn.l2_loss(self.user_bias) + tf.nn.l2_loss(self.item_bias)
        self.l2_regularization += tf.nn.l2_loss(self.global_bias)
        self.loss = self.cross_entropy + l2_reg_lambda * self.l2_regularization


class LMF_DR(object):
    def __init__(self, user_num, item_num, embedding_size, l2_reg_lambda, l2_reg_lambda_il):
        self.user_id = tf.placeholder(tf.int32, [None])
        self.item_id = tf.placeholder(tf.int32, [None])
        self.o = tf.placeholder(tf.float32, [None])
        self.p = tf.placeholder(tf.float32, [None])
        self.y = tf.placeholder(tf.float32, [None])

        initializer = tf.random_normal_initializer()
        self.user_embedding = tf.Variable(initializer([user_num, embedding_size]))
        self.item_embedding = tf.Variable(initializer([item_num, embedding_size]))
        self.user_bias = tf.Variable(initializer([user_num]))
        self.item_bias = tf.Variable(initializer([item_num]))
        self.global_bias = tf.Variable(initializer([1]))
        self.user_embedding_il = tf.Variable(initializer([user_num, embedding_size]))
        self.item_embedding_il = tf.Variable(initializer([item_num, embedding_size]))
        self.user_bias_il = tf.Variable(initializer([user_num]))
        self.item_bias_il = tf.Variable(initializer([item_num]))
        self.global_bias_il = tf.Variable(initializer([1]))

        self.user_feature = tf.nn.embedding_lookup(self.user_embedding, self.user_id)
        self.item_feature = tf.nn.embedding_lookup(self.item_embedding, self.item_id)
        self.b_u = tf.nn.embedding_lookup(self.user_bias, self.user_id)
        self.b_i = tf.nn.embedding_lookup(self.item_bias, self.item_id)
        self.prediction = tf.reduce_sum(tf.multiply(self.user_feature, self.item_feature), 1)
        self.prediction += self.b_u + self.b_i + self.global_bias
        self.cvr = tf.squeeze(tf.sigmoid(tf.expand_dims(self.prediction, 1)))

        self.user_feature_il = tf.nn.embedding_lookup(self.user_embedding_il, self.user_id)
        self.item_feature_il = tf.nn.embedding_lookup(self.item_embedding_il, self.item_id)
        self.b_u_il = tf.nn.embedding_lookup(self.user_bias_il, self.user_id)
        self.b_i_il = tf.nn.embedding_lookup(self.item_bias_il, self.item_id)
        self.prediction_il = tf.reduce_sum(tf.multiply(self.user_feature_il, self.item_feature_il), 1)
        self.prediction_il += self.b_u_il + self.b_i_il + self.global_bias_il
        self.cvr_il = tf.squeeze(tf.sigmoid(tf.expand_dims(self.prediction_il, 1)))
        ones = tf.ones_like(self.cvr_il)
        zeros = tf.zeros_like(self.cvr_il)
        label_unit_step = tf.stop_gradient(tf.where(self.cvr_il < 0.5, x=zeros, y=ones))
        label = tf.stop_gradient(self.cvr_il)

        # Imputation learning
        self.cross_entropy_il = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction_il, labels=self.y)
        self.cross_entropy_il = tf.div(self.cross_entropy_il, self.p)
        self.cross_entropy_mrdr = tf.multiply(self.cross_entropy_il, tf.div(1.0-self.p, self.p))
        self.l2_regularization_il = tf.nn.l2_loss(self.user_embedding_il) + tf.nn.l2_loss(self.item_embedding_il)
        self.l2_regularization_il += tf.nn.l2_loss(self.user_bias_il) + tf.nn.l2_loss(self.item_bias_il)
        self.l2_regularization_il += tf.nn.l2_loss(self.global_bias_il)
        self.loss_il = tf.reduce_mean(self.cross_entropy_il) + l2_reg_lambda_il * self.l2_regularization_il
        self.loss_il_mrdr = tf.reduce_mean(self.cross_entropy_mrdr) + l2_reg_lambda_il * self.l2_regularization_il

        # Prediction learning with probability values
        self.error_il = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=label)
        self.error = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        self.dr_loss = tf.multiply(self.error-self.error_il, tf.div(self.o, self.p)) + self.error_il
        self.l2_regularization = tf.nn.l2_loss(self.user_embedding) + tf.nn.l2_loss(self.item_embedding)
        self.l2_regularization += tf.nn.l2_loss(self.user_bias) + tf.nn.l2_loss(self.item_bias)
        self.l2_regularization += tf.nn.l2_loss(self.global_bias)
        self.loss = tf.reduce_mean(self.dr_loss) + l2_reg_lambda * self.l2_regularization

        # Prediction learning with binary values
        self.error_il_binary = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=label_unit_step)
        self.dr_loss_binary = tf.multiply(self.error - self.error_il_binary, tf.div(self.o, self.p)) + self.error_il_binary
        self.loss_binary = tf.reduce_mean(self.dr_loss_binary) + l2_reg_lambda * self.l2_regularization

        # For evaluation
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.prediction, labels=self.y)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy)

        # For plot
        self.cross_entropy_mae = tf.reduce_mean(tf.abs(self.error_il - self.error))
