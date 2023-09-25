import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import time
from numpy import linalg as LA


def vf_recognise(VF_LIVE, VF_DB):

    dotscore=np.multiply(VF_LIVE.u, VF_DB.u) + np.multiply(VF_LIVE.v, VF_DB.v)

    LIVE_MAG=(np.multiply(VF_LIVE.u, VF_LIVE.u)+np.multiply(VF_LIVE.v, VF_LIVE.v))
    DB_MAG=(np.multiply(VF_DB.u, VF_DB.u)+np.multiply(VF_DB.v, VF_DB.v))

    denom=np.sqrt(np.multiply(LIVE_MAG.sum(),DB_MAG.sum()))

    omega=np.abs(((DB_MAG-LIVE_MAG)/LIVE_MAG))

    scoremat=np.multiply(omega,dotscore)

    score=dotscore.sum()/denom

    #print(omega)

    return score

def vf_recognise3D(VF_LIVE, VF_DB, behind_plane):

    dotscore=np.multiply(VF_LIVE.u[behind_plane], VF_DB.u[behind_plane]) + np.multiply(VF_LIVE.v[behind_plane], VF_DB.v[behind_plane]) + np.multiply(VF_LIVE.w[behind_plane], VF_DB.w[behind_plane])

    LIVE_MAG=(np.multiply(VF_LIVE.u[behind_plane], VF_LIVE.u[behind_plane])+np.multiply(VF_LIVE.v[behind_plane], VF_LIVE.v[behind_plane])+np.multiply(VF_LIVE.w[behind_plane], VF_LIVE.w[behind_plane]))
    DB_MAG=(np.multiply(VF_DB.u[behind_plane], VF_DB.u[behind_plane])+np.multiply(VF_DB.v[behind_plane], VF_DB.v[behind_plane])+np.multiply(VF_DB.w[behind_plane], VF_DB.w[behind_plane]))
    
    denom=np.sqrt(np.multiply(LIVE_MAG.sum(),DB_MAG.sum()))

    if denom==0:
        score=0
    else:
        score=(dotscore.sum()/denom)

    #omega=np.abs(1-((DB_MAG-LIVE_MAG)/LIVE_MAG))
    # scoremat=np.multiply(omega,dotscore)
    # score=scoremat.sum()

    return score

