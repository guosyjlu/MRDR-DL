"""
Preprocessing the real-world dataset, Yahoo and Coat
"""
import pickle
import numpy as np


class Data(object):
    def __init__(self, name, random_seed=1024):
        np.random.seed(random_seed)
        # train_matrix = [user, item, rating] * N
        # test_matrix = [user, item, rating] * M
        if name == "coat":
            train_matrix = np.loadtxt("./data/coat/train.ascii", dtype=int)
            test_matrix = np.loadtxt("./data/coat/test.ascii", dtype=int)
            user, item = np.where(train_matrix)
            rating = train_matrix[user, item]
            train_matrix = np.stack([user, item, rating], axis=1)
            user, item = np.where(test_matrix)
            rating = test_matrix[user, item]
            test_matrix = np.stack([user, item, rating], axis=1)
            self.test_user_item = 16
        elif name == "yahoo":
            train_matrix = np.loadtxt("./data/yahoo/ydata-ymusic-rating-study-v1_0-train.txt", dtype=int)
            test_matrix = np.loadtxt("./data/yahoo/ydata-ymusic-rating-study-v1_0-test.txt", dtype=int)
            train_matrix[:, :-1] -= 1
            test_matrix[:, :-1] -= 1
            self.test_user_item = 10
        else:
            raise Exception("Only support coat and yahoo.")
        self.user_num = np.max(train_matrix[:, 0])+1
        self.item_num = np.max(train_matrix[:, 1])+1
        self.train_num = train_matrix.shape[0]
        self.test_num = test_matrix.shape[0]

        # Get unobserved data
        all_matrix = np.array([[x0, y0] for x0 in np.arange(self.user_num) for y0 in np.arange(self.item_num)])
        print(all_matrix.shape)
        missing_matrix = np.array(list(
                set([tuple(x) for x in all_matrix]) - set([tuple(x) for x in train_matrix[:, :2]])
        ))
        print(missing_matrix.shape)
        print(missing_matrix)
        self.missing_num = missing_matrix.shape[0]

        # Split the original training set into the training set (90%) and validation set (10%)
        index = np.random.permutation(self.train_num)
        self.valid_num = int(0.1 * self.train_num)
        self.train_num = self.train_num - self.valid_num
        valid_matrix = train_matrix[index][:self.valid_num]
        train_matrix = train_matrix[index][self.valid_num:]
        assert self.train_num == train_matrix.shape[0]

        self.user_train, self.item_train, self.convert_train = self.split_matrix(train_matrix)
        self.user_valid, self.item_valid, self.convert_valid = self.split_matrix(valid_matrix)
        self.user_test, self.item_test, self.convert_test = self.split_matrix(test_matrix)
        self.user_missing, self.item_missing, _ = self.split_matrix(missing_matrix)
        self.ctr_train = None

    @staticmethod
    def split_matrix(matrix):
        user = matrix[:, 0]
        item = matrix[:, 1]
        if matrix.shape[1] == 3:
            convert = np.array(matrix[:, 2] > 3, dtype=int)
        else:
            convert = 0
        return user, item, convert

    def get_training_data(self, sample_ratio=2):
        """
        Generate data for training.
        :param sample_ratio: The ratio of unobserved data to observed data.
        sample ratio = -1 : All data
        sample ratio = 0 : observed data only

        :return: (user_id, item_id, click, convert)
        """
        if sample_ratio == -1:
            user = np.append(self.user_train, self.user_missing)
            item = np.append(self.item_train, self.item_missing)
            click = np.array([1]*self.train_num + [0]*self.missing_num)
            convert = np.append(self.convert_train, np.array([0]*self.missing_num))
        elif sample_ratio == 0:
            user = self.user_train
            item = self.item_train
            click = np.array([1]*self.train_num)
            convert = self.convert_train
        else:
            missing_num = min(self.missing_num, self.train_num*sample_ratio)
            index = np.random.choice(self.missing_num, missing_num, replace=False)
            user = np.append(self.user_train, self.user_missing[index])
            item = np.append(self.item_train, self.item_missing[index])
            click = np.array([1] * self.train_num + [0] * missing_num)
            convert = np.append(self.convert_train, np.array([0] * missing_num))
        return user, item, click, convert

    def get_valid_data(self):
        return self.user_valid, self.item_valid, np.array([1]*self.valid_num), self.convert_valid

    def get_test_data(self):
        return self.user_test, self.item_test, np.array([1]*self.test_num), self.convert_test


if __name__ == '__main__':
    coat = Data("coat")
    file = open("data/coat.data", "wb")
    pickle.dump(coat, file)
    file.close()
    print(coat.get_training_data(sample_ratio=-1))
    print(coat.get_training_data(sample_ratio=0))
    print(coat.get_training_data())


