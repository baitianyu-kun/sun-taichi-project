import matplotlib
import numpy as np
import scienceplots
import matplotlib.pyplot as plt

matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use(['science', 'notebook', 'ieee', 'grid', 'std-colors','cjk-sc-font'])
# PR曲线
plt.figure(figsize=(7, 7), dpi=200)
plt.xlabel('Recall')
plt.ylabel('Precision')

SVM_Data = np.load('./svm_result/svm_pr_curve.npz')
KNN_Data = np.load('./knn_3_result/knn_3_pr_curve.npz')
NaiveBayes_Data = np.load('./naive_bayes_result/naive_bayes_pr_curve.npz')
DecisionTree_Data = np.load('./tree_result/tree_pr_curve.npz')
Logit_Reg_Data = np.load('./logistic_regression_result/logistic_regression_pr_curve.npz')

plt.step(SVM_Data['recall_micro'], SVM_Data['precision_micro'], where='post',
         label='SVM : AP={0:0.2f}'.format(SVM_Data['ap_all_micro']))
plt.step(KNN_Data['recall_micro'], KNN_Data['precision_micro'], where='post',
         label='KNN : AP={0:0.2f}'.format(KNN_Data['ap_all_micro']))
plt.step(NaiveBayes_Data['recall_micro'], NaiveBayes_Data['precision_micro'], where='post',
         label='朴素贝叶斯 : AP={0:0.2f}'.format(NaiveBayes_Data['ap_all_micro']))
plt.step(DecisionTree_Data['recall_micro'], DecisionTree_Data['precision_micro'], where='post',
         label='决策树 : AP={0:0.2f}'.format(DecisionTree_Data['ap_all_micro']))
plt.step(Logit_Reg_Data['recall_micro'], Logit_Reg_Data['precision_micro'], where='post',
         label='逻辑回归 : AP={0:0.2f}'.format(Logit_Reg_Data['ap_all_micro']))

plt.legend()
plt.title('Precision-Recall')
plt.savefig('./pr_all_curve.jpg')
plt.show()
