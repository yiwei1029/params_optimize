import numpy as np
from scipy.optimize import minimize

def cal_v(rn,rg,x):
    '''
    函数体内写计算v的表达式
    :param rn: 待优化系数
    :param rg: 待优化系数
    :param x: x样本
    :return: v
    '''
    return rn+rg*x
def overlay_funcs(x_arr): #映射x序列到v
    return lambda rn,rg:[(lambda rn,rg: cal_v(rn,rg,x))(rn,rg) for x in x_arr]
def loss_func(x_arr,y): #合成函数和样本y的差距损失

   return lambda args : sum(((np.array([overlay_funcs(x_arr)(args[i],args[i+1])
                                        for i in range(0,len(args),2)])
                         .sum(0))-y)**2)
if __name__ == '__main__':
    # print(overlay_funcs([1,1])(1,2))
    # print(loss_func([1,2,3,4],[0,0,0,1])([1,2,3,4]))
    args=np.array([1,2,3,4,5,6]) #rn,rg 初始值
    x_arr = [1,2,3,4]
    y=np.array([5,6,7,8],dtype=np.float32)
    res = minimize(loss_func(x_arr,y), args, method='SLSQP')
    print(res.success) #优化是否成功
    print(res.x) #最后得到的rn, rg 值
    print(res.fun) #最终损失函数的值