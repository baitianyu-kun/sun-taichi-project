import numpy as np
from sklearn import linear_model, preprocessing
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier

data = np.loadtxt('./many_gesture_value.txt', delimiter=',')
degrees = data[:, 0:11]
target = data[:, 11].astype(int)
test_vector = [[38.55555, 105, 61, 63, 103, 25, 18, 173, 179, 151, 16],
               [14, 11, 173, 179, 160, 13, 12, 172, 179, 163, 9]]
x, y = degrees, target
# 线性回归
from sklearn import linear_model

linear = linear_model.LinearRegression()
linear.fit(x, y)
print("linear's score: ", linear.score(x, y))
print("w:", linear.coef_)
print("b:", linear.intercept_)
print("predict: ", linear.predict(test_vector))

# 逻辑回归
LR = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000)
LR.fit(x, y)
print("LogProbably: ", LR.predict_proba(test_vector))
print("LogProbablyResult: ", np.argmax(LR.predict_proba(test_vector), axis=1))
print("LogisticRegression:", LR.predict(test_vector))

# 特征重要程度
model = ExtraTreesClassifier()
model.fit(x, y)
# display the relative importance of each attribute
print('Feature_importance:', model.feature_importances_)

# 决策树
from sklearn import tree

TR = tree.DecisionTreeClassifier(criterion='entropy')
TR.fit(x, y)
print("DecisionTree:", TR.predict(test_vector))

# 支持向量机
from sklearn import svm

SV = svm.SVC()
SV.fit(x, y)
print("svm:", SV.predict(test_vector))
print("SV Score:", SV.score(x, y))

# 朴素贝叶斯
from sklearn import naive_bayes

NB = naive_bayes.GaussianNB()
NB.fit(x, y)
print("naive_bayes:", NB.predict(test_vector))

# K近邻
from sklearn import neighbors

KNN = neighbors.KNeighborsClassifier(n_neighbors=3)
KNN.fit(x, y)
print("KNeighbors:", KNN.predict(test_vector))
