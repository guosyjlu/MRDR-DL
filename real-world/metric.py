"""
Metrics for evaluating the performance.
"""
import pickle
import numpy as np
from preprocessor import Data


def type_confirm(data: Data) -> Data:
    return data


def dcg_at_k(obj: Data, score, eval=True):
    """
    Calculate the DCG score
    :param obj: A instance of Data.
    :param score: The predicted CVR score for evaluation.
    :return: A list of DCG scores: [@2, @4, @6]
    """
    score = np.array(score)
    user, item, click, convert = obj.get_test_data()
    user_num = np.max(user) + 1
    convert = convert.reshape([user_num, obj.test_user_item])
    score = score.reshape([user_num, obj.test_user_item])
    ###
    convert_sum = np.sum(convert, axis=1)
    index = np.where(convert_sum > 0)
    convert = convert[index]
    score = score[index]
    user_num = convert.shape[0]
    ###
    index = np.argsort(-score)  # Descending order
    convert = np.array([convert[i][index[i]] for i in range(user_num)])
    dcg_k = []
    if eval:
        for k in [2, 4, 6]:
            convert_k = convert[:, :k]
            order = np.log2(np.arange(k) + 2)
            dcg = np.divide(convert_k, order)
            dcg_k.append(np.sum(dcg) / user_num)
    else:
        for k in [1, 2, 3, 4, 5, 6]:
            convert_k = convert[:, :k]
            order = np.log2(np.arange(k) + 2)
            dcg = np.divide(convert_k, order)
            dcg_k.append(np.sum(dcg) / user_num)
    return np.array(dcg_k)


def recall_at_k(obj: Data, score, eval=True):
    """
        Calculate the Recall score
        :param obj: A instance of Data.
        :param score: The predicted CVR score for evaluation.
        :return: A list of Recall scores: [@2, @4, @6]
        """
    score = np.array(score)
    user, item, click, convert = obj.get_test_data()
    user_num = np.max(user) + 1
    convert = convert.reshape([user_num, obj.test_user_item])
    score = score.reshape([user_num, obj.test_user_item])
    ###
    convert_sum = np.sum(convert, axis=1)
    index = np.where(convert_sum > 0)
    convert = convert[index]
    score = score[index]
    user_num = convert.shape[0]
    ###
    index = np.argsort(-score)  # Descending order
    convert = np.array([convert[i][index[i]] for i in range(user_num)])
    recall_k = []
    if eval:
        for k in [2, 4, 6]:
            convert_k = convert[:, :k]
            recall_k.append(np.sum(convert_k) / user_num)
    else:
        for k in [1, 2, 3, 4, 5, 6]:
            convert_k = convert[:, :k]
            recall_k.append(np.sum(convert_k) / user_num)
    return np.array(recall_k)


if __name__ == '__main__':
    file = open("data/coat.data", "rb")
    coat = pickle.load(file)
    file.close()
    list = dcg_at_k(coat, np.zeros([coat.user_num * coat.test_user_item]))
    print(list)
    list = recall_at_k(coat, np.zeros([coat.user_num * coat.test_user_item]))
    print(list)
