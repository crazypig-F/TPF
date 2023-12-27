import random

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self, X, y, test_size=0.3, seed=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.seed = seed
        self.model = None

    def split_my(self):
        random.seed(self.seed)
        n = self.X.shape[0] // 3
        tn = int(n * (1 - self.test_size))
        arr = list(range(n))
        random.shuffle(arr)
        random.shuffle(arr)
        train = []
        test = []
        for i in arr[:tn]:
            train.append(i * 3)
            train.append(i * 3 + 1)
            train.append(i * 3 + 2)
        for i in arr[tn:]:
            test.append(i * 3)
            test.append(i * 3 + 1)
            test.append(i * 3 + 2)
        return pd.DataFrame(self.X).iloc[train, :], pd.DataFrame(self.X).iloc[test, :], [self.y[i] for i in train], [
            self.y[i] for i in test]

    def cv_my(self, num):
        ss = StandardScaler()
        X = ss.fit_transform(self.X)
        random.seed(self.seed)
        n = self.X.shape[0] // 3
        test_num = n // num
        arr = list(range(n))
        random.shuffle(arr)
        res = []
        for fold in range(num):
            train = []
            test = []
            for i in arr[(fold + 1) * test_num:] + arr[: fold * test_num]:
                train.append(i * 3)
                train.append(i * 3 + 1)
                train.append(i * 3 + 2)
            for i in arr[fold * test_num: (fold + 1) * test_num]:
                test.append(i * 3)
                test.append(i * 3 + 1)
                test.append(i * 3 + 2)
            res.append((X[train, :], X[test, :], [self.y[i] for i in train], [self.y[i] for i in test]))
        return res

    def split(self):
        ss = StandardScaler()
        self.X = ss.fit_transform(self.X)
        return train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.seed)

    def cv(self, kf_num):
        ss = StandardScaler()
        self.X = ss.fit_transform(self.X)
        kf = KFold(n_splits=kf_num, shuffle=True, random_state=self.seed)
        res = []
        for train_index, test_index in kf.split(self.X):
            train_X, train_y = self.X[train_index], self.y[train_index]
            test_X, test_y = self.X[test_index], self.y[test_index]
            res.append((train_X, test_X, train_y, test_y))
        return res
