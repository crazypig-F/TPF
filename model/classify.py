import math

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from analyse.ML import plot
from analyse.ML.model import Model


class ClassifyModel(Model):
    def __init__(self, X, y, test_size=0.3, seed=42):
        self.X = X
        self.y = y
        super().__init__(self.X, self.y, test_size=test_size, seed=seed)

    def train(self):
        model = RandomForestClassifier(n_estimators=50, random_state=self.seed)
        # model = SVC(random_state=self.seed)
        # model = LogisticRegression(random_state=self.seed, max_iter=1000)
        # model = KNeighborsClassifier()
        # X_train, X_test, y_train, y_test = self.split()
        best_s = -math.inf
        score_list = []
        kf_num = 4
        for idx, (X_train, X_test, y_train, y_test) in enumerate(self.cv_my(kf_num)):
            model.fit(X_train, y_train)
            y_p = model.predict(X_test)
            score = sum(y_p == y_test) / len(y_p)
            print(score)
            if score > best_s:
                best_s = score
            score_list.append(score)
            plot.plot_cm(idx, y_test, y_p)
            plot.plot_feature(idx, model, self.X.columns)
        score_sum = np.array(score_list)
        return score_sum.mean(), score_sum.std()
