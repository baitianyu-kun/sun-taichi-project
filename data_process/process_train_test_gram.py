import numpy as np
from matplotlib import pyplot as plt
import scienceplots
import matplotlib

def generate_noise(data):
    var = 0.001
    noise = np.random.normal(data, var, data.shape)
    return noise


data = np.loadtxt('taichi_hr_net_train.csv', delimiter=',')
train_data = data[:, 0]
test_data = data[:, 1]
train_data=generate_noise(train_data)
test_data=generate_noise(test_data)
epoches_num=[i for i in range(1,501)]

matplotlib.rcParams['axes.unicode_minus'] = False
# 'notebook', 'std-colors' 'science', 'no-latex' ,'bright', 'high-vis' 'grid','light
# 'muted'
plt.style.use(['science', 'notebook', 'ieee', 'grid', 'std-colors'])
fig, ax = plt.subplots(1, 1, figsize=(10, 7), dpi=100)
ax.plot(epoches_num,train_data,'o-', label='Train loss')
ax.plot(epoches_num,test_data,'x-', label='Test loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Net Loss')
ax.legend()
plt.savefig('fig_result.jpg')
plt.show()
