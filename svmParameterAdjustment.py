# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 8:36
# @Author  : Youpeng Li
# @Site    : 
# @File    : svmParameterAdjustment.py
# @Software: PyCharm

import sys
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

# SVM Classifier using cross validation and gridsearch
def svm_cross_validation(train_x, train_y):
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC

    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    clf = GridSearchCV(model, param_grid, cv=5, scoring='precision_macro')
    clf.fit(train_x, train_y)
    best_parameters = clf.best_estimator_.get_params()

    for para, val in list(best_parameters.items()):
        print(para, val)

    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

def main():
    # 导入数据集，分成train和test集
    digits = datasets.load_digits()

    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)  # 用训练集训练这个学习器clf
        clf.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
        print(clf.best_params_)
        print()

        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        # 看一下具体的参数间不同数值的组合后得到的分数
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std, params))
            # print()
            # print("Detailed classification report:")
            # print("The model is trained on the full development set.")
            # print("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        # 打印在测试集上的预测结果与真实值的分数
        print(classification_report(y_true, y_pred))

    svm_cross_validation(X_train, y_train)

if __name__ == "__main__":
    sys.exit(main())