# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
plt.style.use('paper.mplstyle')

from offest_env import offest_env
from DDPG import DDPG

#%%
env = offset_env(T=1/12, sigma=0.5, kappa=0.03, eta = 0.05, xi=0.1,
                 c=0.25, S0=2.5, R=5, pen=2.5)

ddpg = DDPG(env, I_max = 10,
            gamma = 0.999, 
            lr=1e-3,
            name="test" )
 
#%%    
ddpg.train(n_plot=200, n_iter_Q=5, n_iter_pi=5)