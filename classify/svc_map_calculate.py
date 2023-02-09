import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
# 我们使用OneVsRestClassifier进行多标签预测
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

data = np.loadtxt('./many_gesture_value_svc_ap_100iter.txt', delimiter=',')
degrees = data[:, 0:11]
target = data[:, 11].astype(int)
X, y = degrees, target

# 使用label_binarize让数据成为类似多标签的设置
Y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
n_classes = Y.shape[1]

# 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.7)

# 运行分类器
classifier = OneVsRestClassifier(svm.SVC())
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

plt.figure("P-R Curve")
plt.title('Precision/Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall[1],precision[1])
plt.show()
print(average_precision)
