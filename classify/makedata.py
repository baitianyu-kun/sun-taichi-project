import numpy as np

if __name__ == '__main__':
    # 每个动作中的每个角度,分别减去0到20之间的随机数,重复50次
    old_data_path = 'gesture_value_no_quanshen.txt'
    new_data_path = 'many_gesture_value_svc_ap_100iter_no_quanshen.txt'
    old_data = np.loadtxt(old_data_path, delimiter=',')
    old_data_degrees = old_data[:, 0:11]
    old_data_targets = old_data[:, 11].astype(int)
    max_iter = 200
    new_data = []
    for old_data_index in range(len(old_data_degrees)):
        old_data_degree = old_data_degrees[old_data_index]
        old_data_target = old_data_targets[old_data_index]
        for iter in range(max_iter):
            # [18 18 13 16 15 13 11  9 19 12 10] 为每个角度生成随机噪声
            noise_plus = np.random.randint(low=0, high=10, size=11)
            # [ -4 -12 -12  -2  -9 -20  -3  -3 -10  -5 -12]
            noise_miner = np.random.randint(low=-100, high=50, size=11)
            old_data_degree_plus = old_data_degree + noise_plus
            # [[角度...,label],[],[]....] 把label和角度放到同一行，方便后面存文件
            new_data.append(np.hstack((old_data_degree_plus, old_data_target)))
            old_data_degree_miner = old_data_degree + noise_miner
            # [[角度...,label],[],[]....] 把label和角度放到同一行，方便后面存文件
            new_data.append(np.hstack((old_data_degree_miner, old_data_target)))
    new_data = np.asarray(new_data)
    np.savetxt(new_data_path, new_data, delimiter=',')
