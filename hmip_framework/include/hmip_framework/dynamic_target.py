#!/usr/bin/env python
import numpy as np
from hmip_framework import vf3D_class as vf3D
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA
from scipy.optimize import least_squares
from decimal import Decimal
from scipy.optimize import minimize


def dynamicVF(wp_x, wp_y, targ_x, targ_y, VF, beta):
    
    v_x=targ_x-wp_x
    v_y=targ_y-wp_y

    distancemesh=VF.calc_distances(wp_x, wp_y)

    alpha=VF.calc_alpha_sigmoid(distancemesh, beta)

    u,v=VF.updateVF(v_x, v_y, alpha)

    return VF

def dynamicVF3D(trajloc, trajvel, trajtime, targ_x, targ_y, targ_z, VF, beta):

    wp_x=trajloc[0,0]
    wp_y=trajloc[0,1]
    wp_z=trajloc[0,2]
    
    v_x=targ_x-wp_x
    v_y=targ_y-wp_y
    v_z=targ_z-wp_z

    distancemesh=VF.calc_distances(wp_x, wp_y, wp_z)

    alpha=VF.calc_alpha_sigmoid(distancemesh, beta)

    u,v,w=VF.updateDBVF(v_x, v_y, v_z, wp_x, wp_y, wp_z, alpha)

    return VF 

def MJT3D(trajloc, trajvel, trajtime,targ_x, targ_y, targ_z, VF, beta):
    
    x0=np.reshape(trajloc[0],(3,1))
    xa=np.array(([[targ_x], [targ_y], [targ_z]]))

    def x_bar(x0, xa, t, tf):
        tau = t / tf
        return x0 + (xa - x0) * (10*tau**3 + 15*tau**4 + 6*tau**5)

    def C(tf, x0, xa, t_values, xk_values):
        x_bar_values = [x_bar(x0, xa, t, tf) for t in t_values]
        errors = np.array(x_bar_values) - np.array(xk_values)
        return LA.norm(errors)**2
    
    t_values=trajtime-trajtime[0]
    xk_values=np.transpose(trajloc)

    result = minimize(lambda tf: C(tf[0], x0, xa, t_values, xk_values), [1.0])
    # print("Optimal tf:", result.x[0])
    # print('time :',t_values[-1])

    tf_optimal=result.x[0]

    timearray=np.linspace(t_values[-1],tf_optimal , 20)

    for k in timearray:
        waypoints=x_bar(x0, xa, k, tf_optimal)
        tau=k/tf_optimal
        velocities=(xa - x0) * (30*tau**2 + 45*tau**3 + 30*tau**4)

        distancemesh=VF.calc_distances(waypoints[0], waypoints[1], waypoints[2])

        alpha=VF.calc_alpha_sigmoid(distancemesh, beta)

        u,v,w=VF.updateDBVF(velocities[0], velocities[1], velocities[2], waypoints[0], waypoints[1], waypoints[2], alpha)

    
    #print('x_0', x_0)
    # print('e_x_a', e_x_a)
    # def MJT(t, tf):
    #     """MJY function"""
    #     t = np.array(t)  # Ensure t is a numpy array
       
    #     #return x_0 + ((10*(t/tf)**3) + (15*(t/tf)**4) + (6*(t/tf)**5))
    #     return x_0 + np.multiply((e_x_a-x_0), ((10*(t/tf)**3) + (15*(t/tf)**4) + (6*(t/tf)**5)))
    # ts=trajtime-trajtime[0]
    # tf_0=ts[-1]

    # def objective_function(x, x_bar):
    #     diff = x - x_bar
    #     return 0.5 * np.sum(np.linalg.norm(diff, axis=-1)**2)
    
    # result = minimize(objective_function, initial_x, args=(x_bar))

    # if result.success:
    #     optimal_x = result.x.reshape(x_bar.shape)
    #     print("Optimal x values:", optimal_x)
    # else:
    #     print("Optimization failed:", result.message)

    # #fun([tf_0])
    # # print('result :', np.float128(res1.x))
    # # print('time :',tf_0)

    return VF