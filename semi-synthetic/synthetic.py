"""
Experiments on semi-synthetic dataset
"""
import pickle
import numpy as np


def getAccuracy(real, predList):
    returnList = []
    for i in predList:
        returnList.append(np.abs(real-i)/real)
    return np.array(returnList)


file = open("data/synthetic_data", "rb")
ground_truth = pickle.load(file)
one = pickle.load(file)
three = pickle.load(file)
five = pickle.load(file)
skew = pickle.load(file)
crs = pickle.load(file)
file.close()

print(ground_truth)
print(one)
print(three)
print(five)
print(skew)
print(crs)

propensity = np.copy(ground_truth)
p = 0.5
propensity[np.where(propensity == 0.9)] = p ** 1
propensity[np.where(propensity == 0.7)] = p ** 2
propensity[np.where(propensity == 0.5)] = p ** 3
propensity[np.where(propensity == 0.3)] = p ** 4
propensity[np.where(propensity == 0.1)] = p ** 4
res = np.zeros([5, 5])
for i in range(20):
    observation = np.random.binomial(1, propensity)
    ones = np.count_nonzero(observation)
    zeros = observation.shape[0] - ones
    p_o = ones/(ones+zeros)

    ground_truth = np.random.binomial(1, ground_truth)
    o = np.where(observation == 1)
    p_hat = 0.5/propensity + 0.5/p_o
    predList = [one, three, five, skew, crs]
    for j in range(5):
        prediction = predList[j]
        ce = -ground_truth * np.log(prediction) - (1 - ground_truth) * np.log(1 - prediction)

        # DR
        prediction_hat = np.sum(prediction * p_hat * observation) / np.sum(observation * p_hat)
        ce_hat = -prediction_hat * np.log(prediction) - (1 - prediction_hat) * np.log(1 - prediction)

        # MRDR
        prediction_hat = np.sum(prediction * p_hat * p_hat * (1 - 1 / p_hat) * observation) / np.sum(
            p_hat * p_hat * (1 - 1 / p_hat) * observation)
        ce_mrdr = -prediction_hat * np.log(prediction) - (1 - prediction_hat) * np.log(1 - prediction)

        real_ce = np.mean(ce)
        naive_ce = np.mean(ce[o])
        eib_ce = np.mean(ce_hat*(1-observation)+ce*observation)
        ips_ce = np.mean(ce * observation * p_hat)
        dr_ce = np.mean(ce_hat + observation * (ce - ce_hat) * p_hat)
        mrdr_ce = np.mean(ce_mrdr + observation * (ce - ce_mrdr) * p_hat)

        acc = getAccuracy(real_ce, [naive_ce, eib_ce, ips_ce, dr_ce, mrdr_ce])
        res[j] += acc
        print(acc)
print()
print(res/20)
