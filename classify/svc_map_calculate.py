import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
# 我们使用OneVsRestClassifier进行多标签预测
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt

data = np.loadtxt('./many_gesture_value_svc_ap_100iter_no_quanshen.txt', delimiter=',')
degrees = data[:, 0:11]
target = data[:, 11].astype(int)
X, y = degrees, target

# 使用label_binarize让数据成为类似多标签的设置
Y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = Y.shape[1]

# 分割训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.7)

# 运行分类器
classifier = OneVsRestClassifier(svm.SVC())
classifier.fit(X_train, Y_train)
y_score = classifier.decision_function(X_test)
y_predict=classifier.predict(X_test)

# 对每个类别
precision = dict()
recall = dict()
average_precision = dict()
# 函数说明average_precision_score(y_true,y_score,*)  根据预测分数计算平均精确率  (AP)
# f1_score(y_true,y_pred,*[,labels,…])  计算F1值
# fbeta_score(y_true,y_pred,*,beta[,…])  计算F-beta值
# precision_recall_curve(y_true,probas_pred,*)  计算不同概率阈值下的精确率召回率对
# precision_recall_fscore_support(y_true,…)  计算每个类的精确率、召回率、F值和真实值的标签数量
# precision_score(y_true,y_pred,*[,labels,…])  计算精确率
# recall_score(y_true,y_pred,*[,labels,…])  计算召回率
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])


# 一个"微观平均": 共同量化所有类别的分数,同时还有marco分数
precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(Y_test, y_score,
                                                     average="micro")
print('Average precision score, micro-averaged over all classes: {0:0.2f}'
      .format(average_precision["micro"]))

# PR曲线
plt.figure()
plt.step(recall['micro'], precision['micro'], where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
# Precision/Recall Curve P-R Curve
plt.title(
    'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
    .format(average_precision["micro"]))
plt.show()

