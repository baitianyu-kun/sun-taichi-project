# 用各种方式在iris数据集上数据分类

# 载入iris数据集，其中每个特征向量有四个维度，有三种类别
from sklearn import datasets

iris = datasets.load_iris()
print("The iris' target names: ", iris.target_names)
x = iris.data
y = iris.target
print(x)
print(y)

# 待分类的两个样本
test_vector = [[1, -1, 2.6, -2], [0, 0, 7, 0.8]]

# 线性回归
from sklearn import linear_model

linear = linear_model.LinearRegression()
linear.fit(x, y)
print("linear's score: ", linear.score(x, y))
print("w:", linear.coef_)
print("b:", linear.intercept_)
print("predict: ", linear.predict(test_vector))

# 逻辑回归
LR = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
LR.fit(x, y)
print("LogisticRegression:", LR.predict(test_vector))

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
'''
he iris' target names:  ['setosa' 'versicolor' 'virginica']
linear's score:  0.930422367533
w: [-0.10974146 -0.04424045  0.22700138  0.60989412]
b: 0.192083994828
predict:  [-0.50300167  2.26900897]
LogisticRegression: [1 2]
DecisionTree: [1 2]
svm: [2 2]
naive_bayes: [2 2]
KNeighbors: [0 1]
'''
