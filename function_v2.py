import numpy as np
from scipy.optimize import minimize
from scipy import integrate
def cal_v(rn,rg,a,h0,t0,x):
    '''
    函数体内写计算v的表达式
    :param rn: 待优化系数
    :param rg: 待优化系数
    :param x: x样本
    :return: v
    '''
    v =  (a*(1-np.exp(-2*rg*(x-t0)*
                       integrate.quad(
                           lambda u: 1-np.exp(-np.pi/3*rn**4/rg*(x-t0)**3*(1-u)**2*(1+2*u)),0,1)[0]))
            +h0)

    return v
def overlay_funcs(x_arr): #映射x序列到v
    return lambda rn,rg,a,h0,t0:[(lambda rn,rg,a,h0,t0: cal_v(rn,rg,a,h0,t0,x))(rn,rg,a,h0,t0) for x in x_arr]
def loss_func(x_arr,y): #合成函数和样本y的差距损失

   return lambda args : sum(((np.array([overlay_funcs(x_arr)(args[i],args[i+1],args[i+2],args[i+3],args[i+4])
                                        for i in range(0,len(args),5)]).sum(0))
                             -y)**2)
if __name__ == '__main__':
    # print(overlay_funcs([1,1])(1,2))
    # print(loss_func([1,2,3,4],[0,0,0,1])([1,2,3,4]))
    args=np.array([1,2,3,4,5,1,1,1,1,1]) #rn,rg 初始值
    x_arr = [1,2,3,4]
    y=np.array([5,6,7,8],dtype=np.float32)
    res = minimize(loss_func(x_arr,y), args, method='SLSQP')
    print(res.success) #优化是否成功
    print(res.x) #最后得到的rn, rg 值
    print(res.fun) #最终损失函数的值