"""
Using standard matrix factorization to generate the complete rating matrix.
"""
import pickle
import numpy as np
from MF import MF
import tensorflow as tf

matrix = np.loadtxt("./data/u.data", dtype=int)[:, :-1]
user = matrix[:, 0] - 1
item = matrix[:, 1] - 1
rating = matrix[:, 2]
user_num = np.max(user)+1
item_num = np.max(item)+1
print(user_num, item_num)
total_num = user.shape[0]
user_train, item_train, rating_train = user[:int(total_num*0.9)], item[:int(total_num*0.9)], rating[:int(total_num*0.9)]
user_test, item_test, rating_test = user[int(total_num*0.9):], item[int(total_num*0.9):], rating[int(total_num*0.9):]
train_num = user_train.shape[0]

batch_size = 1024
l2_reg_lambda = 1e-3    # Validated by grid-search
mf = MF(user_num=user_num, item_num=item_num, embedding_size=64, l2_reg_lambda=l2_reg_lambda)
train_op = tf.train.AdamOptimizer().minimize(mf.loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
test_dict = {
    mf.user_id: user_test,
    mf.item_id: item_test,
    mf.y: rating_test
}
early_stop = 1
best_mse = 100
epoch = 0
while early_stop < 5:
    epoch += 1
    n_batch = train_num // batch_size
    for batch in range(n_batch):
        feed_dict = {
            mf.user_id: user_train[batch * batch_size:(batch + 1) * batch_size],
            mf.item_id: item_train[batch * batch_size:(batch + 1) * batch_size],
            mf.y: rating_train[batch * batch_size:(batch + 1) * batch_size]
        }
        sess.run(train_op, feed_dict)
    prediction, mse = sess.run([mf.prediction, mf.mse], test_dict)
    if mse < best_mse:
        best_mse = mse
        early_stop = 0
    else:
        early_stop += 1
    print("Epoch:", epoch, "MSE:", mse)

all_matrix = np.array([[x0, y0] for x0 in np.arange(user_num) for y0 in np.arange(item_num)])
user_all = all_matrix[:, 0]
item_all = all_matrix[:, 1]
rating_all = np.zeros(user_all.shape)
feed_dict = {
    mf.user_id: user_all,
    mf.item_id: item_all,
    mf.y: rating_all
}
prediction = sess.run(mf.prediction, feed_dict)

file = open("data/predicted_matrix", "wb")
pickle.dump(prediction, file)
pickle.dump(user_num, file)
pickle.dump(item_num, file)
file.close()
