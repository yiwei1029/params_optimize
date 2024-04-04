import copy
import random
import typing

import numpy as np


def cal_v(rn,rg,x):
    return rn*x+2*rg
def gen_args(n):
    args=[]
    for i in range(n):
        rn = random.random()
        rg= random.random()
        args.extend([rn,rg])
    return args
def overlay_funcs(args):
    '''
    to overlay different velocity  functions
    :param n: number of functions to overlay
    :return: function
    '''
    params = []
    for i in range(0,len(args),2):
        rn,rg = args[i:i+2]
        # v = lambda x: cal_v(rn,rg,x)
        params.append((rn,rg))

    return lambda x:sum([(lambda x: cal_v(rn,rg,x))(x) for rn,rg in params])
def loss_func(args,x,y):
    vf=np.vectorize(overlay_funcs(args))
    y_comp=vf(x)
    return ((y_comp-y)**2).sum()

