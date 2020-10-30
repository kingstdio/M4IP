# 数据处理
import os
import numpy as np
import pickle as pkl
# 绘图
# import seaborn as sns
# import matplotlib.pyplot as plt
# %matplotlib inline
# 各种模型、数据处理方法
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, \
    GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, precision_recall_curve, auc, recall_score,precision_score,f1_score
import logging
import argparse
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

parser = argparse.ArgumentParser()

parser.add_argument('--task', type=str, default='41401')
parser.add_argument('--NUM_FEATURES', type=int, default=50)
parser.add_argument('--LIMIT_LEN', type=int, default=64)
args = parser.parse_args()


class Bagging(object):

    def __init__(self, estimators):
        self.estimator_names = []
        self.estimators = []
        for i in estimators:
            self.estimator_names.append(i[0])
            self.estimators.append(i[1])
        self.clf = LogisticRegression()

    def fit(self, train_x, train_y):
        for i in self.estimators:
            i.fit(train_x, train_y)
        x = np.array([i.predict(train_x) for i in self.estimators]).T
        y = train_y
        self.clf.fit(x, y)

    def predict(self, x):
        x = np.array([i.predict(x) for i in self.estimators]).T
        # print(x)
        return self.clf.predict(x)

    def accuracy(self, x, y):
        s = accuracy_score(y, self.predict(x))
        # print(s)
        return s

    def mse(self, x, y):
        s = mean_squared_error(y, self.predict(x))
        return s

    def auc(self, x, y):
        s = roc_auc_score(y, self.predict(x))
        return s


def load_data(file_path, task, logger):
    logger.info('Loading {} data...'.format(task))
    file_path += task
    x_train = np.load(file_path + '/train_x.npy')
    y_train = np.load(file_path + '/train_y.npy')
    x_test = np.load(file_path + '/test_x.npy')
    y_test = np.load(file_path + '/test_y.npy')
    return x_train, y_train, x_test, y_test



def compute_score(X_train, X_test, Y_train, Y_test, logger):
    logger.info('Computing score')
    folds = 5
    lr = LogisticRegression()

    estimators_range = [200, 250, 300, 350, 400]
    leaf_range = [2, 3, 4, 5, 6]
    param_grid = {'n_estimators': estimators_range, 'min_samples_leaf': leaf_range}
    rf = RandomForestRegressor(n_estimators=400, min_samples_leaf=2)
    gs = GridSearchCV(estimator=rf, param_grid=param_grid, n_jobs=4, cv=folds)
    gs = gs.fit(X_train, Y_train)
    logger.info('RandomForest best_score {} best_params {}'.format(gs.best_score_, gs.best_params_))
    rf = gs.best_estimator_

    estimators_range = [400, 450, 500, 550, 600]
    rate_range = [0.03, 0.035, 0.04, 0.045, 0.05]
    depth_range = [2, 3, 4, 5]
    param_grid = {'n_estimators': estimators_range, 'learning_rate': rate_range, 'max_depth': depth_range}
    gbdt = GradientBoostingRegressor(n_estimators=450, learning_rate=0.04, max_depth=3)
    gs = GridSearchCV(estimator=gbdt, param_grid=param_grid, n_jobs=4, cv=folds)
    gs = gs.fit(X_train, Y_train)
    logger.info('GradientBoosting best_score {} best_params {}'.format(gs.best_score_, gs.best_params_))
    gbdt = gs.best_estimator_

    estimators_range = [400, 450, 500, 550, 600]
    rate_range = [0.03, 0.035, 0.04, 0.045, 0.05]
    depth_range = [2, 3, 4, 5]
    param_grid = {'n_estimators': estimators_range, 'learning_rate': rate_range, 'max_depth': depth_range}
    xgbGBDT = XGBRegressor(n_estimators=500, learning_rate=0.04, max_depth=4)
    gs = GridSearchCV(estimator=xgbGBDT, param_grid=param_grid, n_jobs=4, cv=folds)
    gs = gs.fit(X_train, Y_train)
    logger.info('XGBoost best_score {} best_params {}'.format(gs.best_score_, gs.best_params_))
    xgbGBDT = gs.best_estimator_

    bag = Bagging([('xgb', xgbGBDT), ('lr', lr), ('gbdt', gbdt), ('rf', rf)])
    score = 0
    num_test = 0.20
    for i in range(0, folds):
        train_x, cv_x, train_y, cv_y = train_test_split(X_train, Y_train, test_size=num_test)
        bag.fit(train_x, train_y)
        # Y_test = bag.predict(X_test)
        mse_xgb = round(bag.mse(cv_x, cv_y) * 100, 4)
        score += mse_xgb
    logger.info('Dev Acc {}'.format(score / folds))

    # Predict
    bag.fit(X_train, Y_train)
    mse = bag.mse(X_test, Y_test)
    return mse


def cal_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_score)
    (precisions, recalls, thresholds) = precision_recall_curve(y_true, y_score)
    prc = auc(recalls, precisions)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return [acc, precision, recall, f1, roc, prc]


def compute_label(X_train, X_test, Y_train, Y_test, logger, path, task):
    logger.info('Computing {}...'.format(task))
    if task == '5849' or '25000':
        job = 8
    else:
        job = 4
    metrics = []
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred = lr.predict(X_test)
    Y_score = lr.predict_proba(X_test)[:,1]
    lr_me = cal_metrics(Y_test, Y_pred, Y_score)
    metrics.append(lr_me)
    logger.info('LR - {}'.format(lr_me))
    del lr, Y_pred, Y_score

    svm = SVC(kernel='rbf',random_state=0,gamma=0.2,C=1.0, probability= True)
    svm.fit(X_train, Y_train)
    Y_pred = svm.predict(X_test)
    Y_score = svm.predict_proba(X_test)[:,1]
    svm_me = cal_metrics(Y_test, Y_pred, Y_score)
    metrics.append(svm_me)
    logger.info('SVM - {}'.format(svm_me))
    del svm, Y_pred, Y_score


    dt = tree.DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    Y_pred = dt.predict(X_test)
    Y_score = dt.predict_proba(X_test)[:,1]
    dt_me = cal_metrics(Y_test, Y_pred, Y_score)
    metrics.append(dt_me)
    logger.info('DT - {}'.format(dt_me))
    del dt, Y_pred, Y_score



    rf = RandomForestClassifier(n_jobs=job)
    rf.fit(X_train, Y_train)
    Y_pred = rf.predict(X_test)
    Y_score = rf.predict_proba(X_test)[:,1]
    rf_me = cal_metrics(Y_test, Y_pred, Y_score)
    metrics.append(rf_me)
    logger.info('RF - {}'.format(rf_me))
    del rf, Y_pred, Y_score

    gbdt = GradientBoostingClassifier()
    gbdt.fit(X_train, Y_train)
    Y_pred = gbdt.predict(X_test)
    Y_score = gbdt.predict_proba(X_test)[:,1]
    gbdt_me = cal_metrics(Y_test, Y_pred, Y_score)
    logger.info('GradientBoosting -  {}'.format(gbdt_me))
    del gbdt, Y_pred, Y_score

    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    Y_score = knn.predict_proba(X_test)[:, 1]
    knn_me = cal_metrics(Y_test, Y_pred, Y_score)
    logger.info('KNN -  {}'.format(knn_me))
    del knn, Y_pred, Y_score


    # nb = MultinomialNB(alpha=0.01)
    # nb.fit(X_train, Y_train)
    # Y_pred = nb.predict(X_test)
    # Y_score = nb.predict_proba(X_test)[:, 1]
    # nb_me = cal_metrics(Y_test, Y_pred, Y_score)
    # metrics.append(nb_me)
    # logger.info('NaiveBayes - {}'.format(nb_me))
    # del nb, Y_pred, Y_score

    xgbGBDT = XGBClassifier(n_jobs=job)
    xgbGBDT.fit(X_train, Y_train)
    Y_pred = xgbGBDT.predict(X_test)
    Y_score = xgbGBDT.predict_proba(X_test)[:,1]
    xgb_me = cal_metrics(Y_test, Y_pred, Y_score)
    metrics.append(xgb_me)
    logger.info('XGBOOST - {}'.format(xgb_me))
    del xgbGBDT, Y_pred, Y_score

    metrics = np.asarray(metrics, dtype=np.float32)
    path = os.path.join(path, task, 'results.txt')
    np.savetxt(path, metrics, delimiter='\t')

def single_result(file_path, task, logger):
    train_x, train_y, test_x, test_y = load_data(file_path, task, logger)
    compute_label(train_x, test_x, train_y, test_y, logger, file_path, task)


if __name__ == '__main__':
    logger = logging.getLogger('Medical baseline')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_path = 'data/preprocessed_data/baseline/'
    single_result(file_path, args.task, logger)

