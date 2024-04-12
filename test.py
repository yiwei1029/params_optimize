import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import integrate
from config import cons


def cal_v(rn, rg, a, h0, t0, x):
    '''
    函数体内写计算v的表达式
    :param rn: 待优化系数
    :param rg: 待优化系数
    :param x: x样本
    :return: v
    '''
    v = (a * (1 - np.exp(-2 * rg * (x - t0) *
                         integrate.quad(lambda u: 1 - np.exp(
                             -np.pi / 3 * rn ** 4 / rg * (x - t0) ** 3 * (1 - u) ** 2 * (1 + 2 * u)), 0, 1)[0])) + h0)
    return v


def map_func(x_arr):  # 映射x序列到v
    return lambda rn, rg, a, h0, t0: \
        [(lambda rn, rg, a, h0, t0: cal_v(rn, rg, a, h0, t0, x))(rn, rg, a, h0, t0)
         for x in x_arr]


# def loss_func(x_arr, y):  # 合成函数和样本y的差距损失
#
#     return lambda args: sum(
#         ((np.array([map_func(x_arr)(args[i], args[i + 1], args[i + 2], args[i + 3], args[i + 4])
#                     for i in range(0, len(args), 5)]).sum(0))
#          - y) ** 2)
def overlay_funcs(x_arr):
    return lambda args: [map_func(x_arr)(args[i], args[i + 1], args[i + 2], args[i + 3], args[i + 4])
                         for i in range(0, len(args), 5)]


def loss_func(x_arr, y):
    return lambda args: sum((np.array(overlay_funcs(x_arr)(args)).sum(0)
                             - y) ** 2)


def plot_res(args, x_arr, y):
    y_pred = np.array(overlay_funcs(x_arr)(args)).sum(0)
    plt.figure(figsize=(10, 8))
    plt.scatter(x_arr, y_pred, label='predicted')
    plt.scatter(x_arr, y, label='actual')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # print(map_func([1, 2, 3, 4])(1, 2, 3, 4, 5))
    # print(loss_func([1,2,3,4],[1,2,3,4])([1,1,1,1,1]))
    # args = np.array([1, 1, 50, 50, 1, 0.1, 1, 100, 50, 1])  # rn,rg 初始值
    args = np.random.rand(10)
    # print(args)
    arr = np.loadtxt('data.csv', delimiter=',')[:400]
    x_arr = arr[:, 0]
    y = arr[:, 1]
    res = minimize(loss_func(x_arr, y), args, method='SLSQP', constraints=cons)
    print(res.success)  # 优化是否成功
    print(res.x)  # 最后得到的rn, rg 值
    print(res.fun / len(y))  # 最终损失函数的值
    plot_res(res.x, x_arr, y)
