"""
    Baseline: Naive.
"""
import pickle
import numpy as np
import tensorflow as tf
from FM import weightFM
from metric import dcg_at_k, recall_at_k, type_confirm

path = "data/coat.data"
file = open(path, "rb")
obj = type_confirm(pickle.load(file))
file.close()

embedding_size = 64
batch_size = 1024
l2_reg_lambda = 1e-3

mf = weightFM(
    user_num=obj.user_num,
    item_num=obj.item_num,
    embedding_size=embedding_size,
    l2_reg_lambda=l2_reg_lambda
)
prediction_op = tf.train.AdamOptimizer().minimize(mf.loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
early_stop = 0
epoch = 0
best_ce = 10
best_dcg = None
best_recall = None
while early_stop < 100:
    epoch += 1
    user, item, click, convert = obj.get_training_data(sample_ratio=0)
    train_num = user.shape[0]
    p = np.array([1] * train_num)
    index = np.random.permutation(train_num)
    user, item, convert = user[index], item[index], convert[index]
    n_batch = train_num // batch_size + 1
    for batch in range(n_batch):
        feed_dict = {
            mf.user_id: user[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.item_id: item[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.y: convert[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.p: p[batch * batch_size: min((batch + 1) * batch_size, train_num)]
        }
        _ = sess.run([prediction_op], feed_dict=feed_dict)

    # Validation performance
    user, item, click, convert = obj.get_valid_data()
    feed_dict = {
        mf.user_id: user,
        mf.item_id: item,
        mf.y: convert,
        mf.p: [1] * user.shape[0]
    }
    ce = sess.run(mf.cross_entropy, feed_dict=feed_dict)

    # Test performance
    user, item, click, convert = obj.get_test_data()
    feed_dict = {
        mf.user_id: user,
        mf.item_id: item,
        mf.y: convert,
        mf.p: [1] * user.shape[0]
    }
    prediction = sess.run(mf.cvr, feed_dict)
    dcg = dcg_at_k(obj, prediction)
    recall = recall_at_k(obj, prediction)
    print("Epoch:", epoch, "Cross entropy:", ce, "DCG@2,4,6:", dcg, "Recall@2,4,6:", recall)

    # Update early stopping
    if ce < best_ce:
        best_ce = ce
        best_dcg = dcg
        best_recall = recall
        early_stop = 0
    else:
        early_stop += 1
print("Best cross entropy:", best_ce)
print("Best DCG@2,4,6:", best_dcg)
print("Best Recall@2,4,6:", best_recall)
