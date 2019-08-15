# -*- coding: utf-8 -*-
# @Time    : 2019/8/15 16:29
# @Author  : Youpeng Li
# @Site    : 
# @File    : svmForUnbalancedClass.py
# @Software: PyCharm

"""
=================================================
SVM: Separating hyperplane for unbalanced classes
=================================================
Find the optimal separating hyperplane using an SVC for classes that
are unbalanced.
We first find the separating plane with a plain SVC and then plot
(dashed) the separating hyperplane with automatically correction for
unbalanced classes.
.. currentmodule:: sklearn.linear_model
.. note::
    This example will also work by replacing ``SVC(kernel="linear")``
    with ``SGDClassifier(loss="hinge")``. Setting the ``loss`` parameter
    of the :class:`SGDClassifier` equal to ``hinge`` will yield behaviour
    such as that of a SVC with a linear kernel.
    For example try instead of the ``SVC``::
        clf = SGDClassifier(n_iter=100, alpha=0.01)
"""
print(__doc__)

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
import warnings
warnings.filterwarnings('ignore')

def main():
    # we create two clusters of random points
    n_samples_1 = 1000
    n_samples_2 = 100
    centers = [[0.0, 0.0], [2.0, 2.0]]
    clusters_std = [1.5, 0.5]
    # 生成聚类数据
    X, y = make_blobs(n_samples=[n_samples_1, n_samples_2],
                      centers=centers,
                      cluster_std=clusters_std,
                      random_state=0, shuffle=False)

    # fit the model and get the separating hyperplane
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X, y)

    # fit the model and get the separating hyperplane using weighted classes
    wclf = svm.SVC(kernel='linear', class_weight={1: 10})
    wclf.fit(X, y)

    # plot the samples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

    # plot the decision functions for both classifiers
    # plt.plot()实际上会通过plt.gca()获得当前的Axes对象ax，然后再调用ax.plot()方法实现真正的绘图。
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    # vstack返回结果为numpy的数组
    xy = np.vstack([XX.ravel(), YY.ravel()]).T

    # get the separating hyperplane
    Z = clf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

    # get the separating hyperplane for weighted classes
    Z = wclf.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins for weighted classes
    b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

    plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
               loc="upper right")
    plt.show()


if __name__ == "__main__":
    sys.exit(main())