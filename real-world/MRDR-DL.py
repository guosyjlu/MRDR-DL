"""
Enhanced Doubly Robust Learning: More Robust Doubly Robust Estimator with Double Learning.
"""
import pickle
import numpy as np
import tensorflow as tf
from FM import drFM
from metric import dcg_at_k, recall_at_k
from data_util import Dataset, type_confirm

path = "data/coat/coat.data"
file = open(path, "rb")
obj = type_confirm(pickle.load(file))
file.close()

# Hyper-parameters
embedding_size = 64
batch_size = 1024
lr = 0.001
lr_il = 0.001
sr = 4
l2 = 1e-4
l2_il = 1e-4

mf = drFM(
    user_num=obj.user_num,
    item_num=obj.item_num,
    embedding_size=embedding_size,
    l2_reg_lambda=l2,
    l2_reg_lambda_il=l2_il
)
prediction_op = tf.train.AdamOptimizer(lr).minimize(mf.loss)
imputation_op = tf.train.AdamOptimizer(lr_il).minimize(mf.loss_il_mrdr)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
copy = [
    tf.assign(mf.user_embedding_il, mf.user_embedding),
    tf.assign(mf.item_embedding_il, mf.item_embedding),
    tf.assign(mf.user_bias_il, mf.user_bias),
    tf.assign(mf.item_bias_il, mf.item_bias),
    tf.assign(mf.global_bias_il, mf.global_bias)
]

early_stop = 0
epoch = 0
best_ce = 10
best_dcg = None
best_recall = None
while early_stop < 5:
    epoch += 1
    # Copy and train the imputation model
    sess.run(copy)
    user, item, click, convert = obj.get_training_data(sample_ratio=0)
    train_num = user.shape[0]
    p = obj.ctr_train
    index = np.random.permutation(train_num)
    user, item, click, convert, p = user[index], item[index], click[index], convert[index], p[index]
    n_batch = train_num // batch_size + 1
    for batch in range(n_batch):
        feed_dict = {
            mf.user_id: user[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.item_id: item[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.y: convert[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.o: click[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.p: p[batch * batch_size: min((batch + 1) * batch_size, train_num)]
        }
        _ = sess.run([imputation_op], feed_dict=feed_dict)

    # Train the prediction model
    user, item, click, convert = obj.get_training_data(sample_ratio=sr)
    p = np.append(obj.ctr_train, np.array([1] * obj.missing_num))
    index = np.random.choice(user.shape[0], train_num, replace=False)
    user, item, click, convert, p = user[index], item[index], click[index], convert[index], p[index]
    n_batch = train_num // batch_size + 1
    for batch in range(n_batch):
        feed_dict = {
            mf.user_id: user[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.item_id: item[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.y: convert[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.o: click[batch * batch_size: min((batch + 1) * batch_size, train_num)],
            mf.p: p[batch * batch_size: min((batch + 1) * batch_size, train_num)]
        }
        _ = sess.run([prediction_op], feed_dict=feed_dict)

    # Validation performance
    user, item, click, convert = obj.get_valid_data()
    feed_dict = {
        mf.user_id: user,
        mf.item_id: item,
        mf.y: convert,
        mf.o: click,
        mf.p: [1] * user.shape[0]
    }
    ce = sess.run(mf.cross_entropy, feed_dict=feed_dict)

    # Test performance
    user, item, click, convert = obj.get_test_data()
    feed_dict = {
        mf.user_id: user,
        mf.item_id: item,
        mf.y: convert,
        mf.o: click,
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
        best_prediction = prediction
    else:
        early_stop += 1

print("Best cross entropy:", best_ce)
print("Best DCG@2,4,6:", best_dcg)
print("Best Recall@2,4,6:", best_recall)
