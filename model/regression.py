import math

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from model.model import Model
from xgboost import XGBRegressor


def relative_root_mean_squared_error(true, pred):
    rmse = np.sqrt(np.mean((true - pred) ** 2))
    # 计算观测数据的均值
    mean_observed = np.mean(true)
    # 计算相对均方根误差 (RRMSE)
    rrmse = (rmse / mean_observed) * 100
    return rrmse


def calculate_r2(observed, predicted):
    # 计算总平方和
    total_sum_of_squares = np.sum((observed - np.mean(observed)) ** 2)
    # 计算残差平方和
    residual_sum_of_squares = np.sum((observed - predicted) ** 2)
    # 计算R²
    r2 = 1 - (residual_sum_of_squares / total_sum_of_squares)
    return r2


class RegressionModel(Model):
    def __init__(self, X, y, test_size=0.3, seed=42):
        super().__init__(X, y, test_size=test_size, seed=seed)

    def search(self):
        # param_test = {"n_estimators": range(1, 501, 10)}
        param_test = {"max_features": range(1, 6, 1)}
        gsearch1 = GridSearchCV(estimator=RandomForestRegressor(), param_grid=param_test,
                                scoring='r2', cv=4)
        gsearch1.fit(self.X, self.y)
        print(gsearch1.best_params_)
        print("best accuracy:%f" % gsearch1.best_score_)

    def train(self):
        model = RandomForestRegressor(n_estimators=71, random_state=self.seed)
        # model = LinearRegression()
        # model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        # model = XGBRegressor(n_estimators=50, random_state=42)
        # X_train, X_test, y_train, y_test = self.split()
        best_r = None
        best_p = None
        best_s = -math.inf
        score_sum = 0
        kf_num = 4
        score_list = []
        for X_train, X_test, y_train, y_test in self.cv(kf_num):
            model.fit(X_train, y_train)
            y_p = model.predict(X_test)
            score = r2_score(y_test, y_p)
            # rrmse = relative_root_mean_squared_error(y_test, y_p)
            # rrmse_list.append(rrmse)
            # print("rrmse", rrmse)
            score_list.append(score)
            if score > best_s:
                best_r = y_test
                best_p = y_p
                best_s = score
        score_list = np.array(score_list)
        return best_r.to_list(), best_p, score_list.mean(), score_list.std(), best_s
