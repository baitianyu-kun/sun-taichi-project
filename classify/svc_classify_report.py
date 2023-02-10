import json
import os.path

import matplotlib
import scienceplots
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import tree
from sklearn import ensemble
from sklearn import linear_model
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
# 我们使用OneVsRestClassifier进行多标签预测
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd


# return ap and ap_all_micro, y_score 为 decision fun返回的概率，直接用y_predict标签的话可能不准确，毕竟算得是面积
def calculate_ap(n_classes, Y_test, y_score, classes_names):
    ap = dict()
    for i in range(n_classes):
        ap[classes_names[i]] = average_precision_score(Y_test[:, i], y_score[:, i])
    return ap, average_precision_score(Y_test, y_score, average='micro'), \
           average_precision_score(Y_test, y_score, average='macro'), \
           average_precision_score(Y_test, y_score, average='weighted'), \
           average_precision_score(Y_test, y_score, average='samples')


def prepare_data(data_path):
    data = np.loadtxt(data_path, delimiter=',')
    degrees = data[:, 0:11]
    target = data[:, 11].astype(int)
    X, y = degrees, target
    # 使用label_binarize让数据成为类似多标签的设置
    Y = label_binarize(y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    n_classes = Y.shape[1]
    # 分割训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.4)
    return X_train, X_test, Y_train, Y_test, n_classes


def run_classify(X_train, X_test, Y_train, Y_test):
    # 运行分类器
    # naive_bayes.GaussianNB() -> predict_proba
    # neighbors.KNeighborsClassifier(n_neighbors=3)-> predict_proba
    # svm.SVC() -> decision_function
    # tree.DecisionTreeClassifier(criterion='entropy') -> predict_proba
    # ensemble.RandomForestClassifier() -> predict_proba -> AP结果过高
    # linear_model.LogisticRegression(solver='lbfgs', max_iter=1000) -> predict_proba

    classifier = OneVsRestClassifier(neighbors.KNeighborsClassifier(n_neighbors=3))
    classifier.fit(X_train, Y_train)
    # y_score = classifier.decision_function(X_test)
    y_score = classifier.predict_proba(X_test)
    y_predict = classifier.predict(X_test)
    return y_score, y_predict


def draw_PR(recall_micro, precision_micro, ap_all_micro, is_save, save_path):
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 'notebook', 'std-colors' 'science', 'no-latex' ,'bright', 'high-vis' 'grid','light
    # 'muted'
    plt.style.use(['science', 'notebook', 'ieee', 'grid', 'std-colors'])
    # PR曲线
    plt.figure(figsize=(7, 7), dpi=200)
    plt.step(recall_micro, precision_micro, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # plt.ylim([0.0, 1.05])
    # plt.xlim([0.0, 1.0])
    # Precision/Recall Curve P-R Curve
    plt.title(
        'Precision-Recall: AP={0:0.2f}'
        .format(ap_all_micro))
    if is_save:
        plt.savefig(save_path)
    plt.show()


gesture_list = ['更鸡独立', '右蹬腿', '手挥琵琶', '白鹤亮翅', '倒撵猴', '高探马', '上步搬拦捶', '单鞭', '右通背',
                '玉女穿梭']
X_train, X_test, Y_train, Y_test, n_classes = prepare_data('./many_gesture_value_svc_ap_100iter_no_quanshen.txt')
y_score, y_predict = run_classify(X_train, X_test, Y_train, Y_test)
ap, ap_all_micro, ap_all_macro, ap_all_weighted, ap_all_samples = calculate_ap(n_classes, Y_test, y_score, gesture_list)
precision_micro, recall_micro, _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
# draw_PR(recall_micro,precision_micro,ap_all_micro)
report = classification_report(Y_test, y_predict, target_names=gesture_list, output_dict=True)
df_report = pd.DataFrame(report).transpose()
# 先赋值为0，再填充AP数据
df_report['AP'] = 0
for key in ap.keys():
    df_report.loc[key, 'AP'] = ap[key]
df_report.loc['micro avg', 'AP'] = ap_all_micro
df_report.loc['macro avg', 'AP'] = ap_all_macro
df_report.loc['weighted avg', 'AP'] = ap_all_weighted
df_report.loc['samples avg', 'AP'] = ap_all_samples
# save
net = 'knn_3'
if not os.path.exists(f'./{net}_result'):
    os.makedirs(f'./{net}_result')
df_report.to_csv(f'./{net}_result/{net}_report.csv', sep=',', encoding='utf_8_sig')
# 多个数组保存到一个文件，前面是数组名，等号后面是数组
np.savez(f'./{net}_result/{net}_pr_curve.npz', recall_micro=recall_micro, precision_micro=precision_micro,
         ap_all_micro=ap_all_micro)
draw_PR(recall_micro, precision_micro, ap_all_micro, True, f'./{net}_result/{net}_pr.jpg')
