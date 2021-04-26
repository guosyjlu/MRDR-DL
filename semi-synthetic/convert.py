"""
Convert the rating matrix into conversion rate matrix;
Generate the simulated prediction conversion rate matrix.
"""
import pickle
import numpy as np

file = open("data/predicted_matrix", "rb")
prediction = np.array(pickle.load(file), dtype=float)
user_num = pickle.load(file)
item_num = pickle.load(file)
file.close()

# CVR = [0.1, 0.3, 0.5, 0.7, 0.9]
# ratio = [0.53, 0.24, 0.14, 0.06, 0.03] (the same distribution as in Yahoo R3! MAR test set)
total_num = prediction.shape[0]
index = np.argsort(prediction)
index_inverse = np.argsort(index)
prediction = prediction[index]
prediction[:int(total_num*0.53)] = 0.1
prediction[int(total_num*0.53):int(total_num*0.77)] = 0.3
prediction[int(total_num*0.77):int(total_num*0.91)] = 0.5
prediction[int(total_num*0.91):int(total_num*0.98)] = 0.7
prediction[int(total_num*0.98):] = 0.9
ground_truth = prediction[index_inverse]
print(ground_truth[:20])

# Simulated prediction 1 - ONE
# Randomly select n_0.9 0.1, and set 0.1 to 0.9, where n_0.9 denotes the number of the 0.9 in ground_truth
n_0_1 = np.count_nonzero(np.where(ground_truth == 0.1))
n_0_9 = np.count_nonzero(np.where(ground_truth == 0.9))
select = np.random.choice(n_0_1, n_0_9, replace=False)
prediction = ground_truth[index]
prediction[select] = 0.9
one = prediction[index_inverse]

# Simulated prediction 2 - THREE
# Randomly select n_0.9 0.3, and set 0.3 to 0.9, where n_0.9 denotes the number of the 0.9 in ground_truth
n_0_3 = np.count_nonzero(np.where(ground_truth == 0.3))
select = np.random.choice(n_0_3, n_0_9, replace=False)+int(total_num*0.53)
prediction = ground_truth[index]
prediction[select] = 0.9
three = prediction[index_inverse]

# Simulated prediction 3 - FIVE
# Randomly select n_0.9 0.5, and set 0.5 to 0.9, where n_0.9 denotes the number of the 0.9 in ground_truth
n_0_5 = np.count_nonzero(np.where(ground_truth == 0.5))
select = np.random.choice(n_0_5, n_0_9, replace=False)+int(total_num*0.77)
prediction = ground_truth[index]
prediction[select] = 0.9
five = prediction[index_inverse]

# Simulated prediction 4 - SKEW
# r ~ N(\mu=r, \sigma=(1-r)/2), and then r is clipped to [0.1~0.9]
prediction = np.copy(ground_truth)
for i in range(prediction.shape[0]):
    prediction[i] = np.random.normal(loc=prediction[i], scale=(1-prediction[i])/2, size=1)
skew = np.clip(prediction, 0.1, 0.9)

# Simulated prediction 5 - CRS
# r:=0.1, if r>=0.7; r:=0.5, else
prediction = np.copy(ground_truth)
select1 = np.where(prediction >= 0.7)
select2 = np.where(prediction < 0.7)
prediction[select1] = 0.1
prediction[select2] = 0.5
crs = prediction
print(crs[:20])
print(ground_truth[:20])

file = open("data/synthetic_data", "wb")
pickle.dump(ground_truth, file)
pickle.dump(one, file)
pickle.dump(three, file)
pickle.dump(five, file)
pickle.dump(skew, file)
pickle.dump(crs, file)
file.close()
