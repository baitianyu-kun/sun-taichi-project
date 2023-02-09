from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import label_binarize
# 我们使用OneVsRestClassifier进行多标签预测
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

iris = datasets.load_iris()
X = iris.data
y = iris.target

# 增加噪音特征
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# 使用label_binarize让数据成为类似多标签的设置
Y = label_binarize(y, classes=[0, 1, 2])
n_classes = Y.shape[1]

# 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                    random_state=random_state)
# 运行分类器
classifier = OneVsRestClassifier(svm.LinearSVC(random_state=random_state))
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
# 对每个类别
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
                                                                y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

