"""
Estimate the propensity score (i.e., CTR) with Logistic Matrix Factorization.
"""
import pickle
import numpy as np
import tensorflow as tf
from MF import logisticMF
from data_util import Data


def type_confirm(data: Data) -> Data:
    return data


path = "../data/coat.data"
file = open(path, "rb")
obj = type_confirm(pickle.load(file))
file.close()
embedding_size = 64
batch_size = 256
best_l2_reg_lambda = -1
best_ce = 10
best_prediction = None
for l2_reg_lambda in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
    mf = logisticMF(
        user_num=obj.user_num,
        item_num=obj.item_num,
        embedding_size=embedding_size,
        l2_reg_lambda=l2_reg_lambda
    )
    prediction_op = tf.train.AdamOptimizer().minimize(mf.loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    early_stop = 0
    local_best_ce = 10
    local_best_prediction = None
    while early_stop < 5:
        user, item, click, convert = obj.get_training_data(sample_ratio=4)
        train_num = user.shape[0]
        index = np.random.permutation(train_num)
        user, item, click = user[index], item[index], click[index]
        n_batch = train_num // batch_size + 1
        for batch in range(n_batch):
            feed_dict = {
                mf.user_id: user[batch*batch_size: min((batch+1)*batch_size, train_num)],
                mf.item_id: item[batch*batch_size: min((batch+1)*batch_size, train_num)],
                mf.y: click[batch*batch_size: min((batch+1)*batch_size, train_num)]
            }
            _ = sess.run([prediction_op], feed_dict=feed_dict)

        # Validation performance
        user, item, click, convert = obj.get_valid_data()
        feed_dict = {
            mf.user_id: user,
            mf.item_id: item,
            mf.y: click
        }
        ce = sess.run(mf.cross_entropy, feed_dict=feed_dict)
        print("Cross entropy:", ce)

        # Predict the CTR
        user, item, click, convert = obj.get_training_data(sample_ratio=0)
        feed_dict = {
            mf.user_id: user,
            mf.item_id: item,
            mf.y: click
        }
        prediction = sess.run(mf.ctr, feed_dict)

        # Update early stopping
        if ce < local_best_ce:
            local_best_ce = ce
            local_best_prediction = prediction
            early_stop = 0
        else:
            early_stop += 1
    if local_best_ce < best_ce:
        best_ce = local_best_ce
        best_l2_reg_lambda = l2_reg_lambda
        best_prediction = local_best_prediction
    print("Local predicted CTR:", local_best_prediction)
print("Best L2 reg lambda:", best_l2_reg_lambda)
print("Predicted CTR:", best_prediction)
obj.ctr_train = best_prediction

file = open(path, "wb")
pickle.dump(obj, file)
file.close()
