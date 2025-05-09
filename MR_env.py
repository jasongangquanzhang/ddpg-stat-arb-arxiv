# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 11:41:23 2022

@author: sebja
"""

import numpy as np
from tqdm import tqdm 
import pdb
import torch

#
# dS_t = \kappa ( \theta - S_t) dt + \sigma dW_t
#
class MR_env():

    def __init__(self, S_0=1300,
                 theta=1300,
                 kappa=5,
                 sigma=1,
                 dt = 1,
                 T=int(60*60),
                 I_max = 10,
                 lambd = 0.02):
        
        self.S_0 = S_0
        self.theta = theta
        self.sigma = sigma
        self.kappa = kappa
        self.lambd = lambd
        
        self.dt = dt  # time of one step
        self.T = T # total time
        self.N = int(self.T/self.dt)+1 # number of time steps
        
        self.t = torch.linspace(0, self.T, self.N)
        
        self.inv_vol = self.sigma/np.sqrt(2.0*self.kappa)
        self.eff_vol = self.sigma* np.sqrt((1-np.exp(-2*self.kappa*self.dt))/(2*self.kappa))
        
        self.I_max = I_max
        
    def lognormal(self, sigma, mini_batch_size=10):
        return torch.exp(-0.5*sigma**2 + sigma*torch.randn(mini_batch_size))
        
    def Randomize_Start(self, mini_batch_size=10):
        
        S0 = self.S_0 + 3*self.inv_vol*torch.randn(mini_batch_size)
        I0 = self.I_max * (2*torch.rand(mini_batch_size)-1)
        
        return S0, I0
# not even used in the code
    # def Simulate(self,  mini_batch_size=10):

    #     S = torch.zeros((mini_batch_size, self.N)).float()
    #     I = torch.zeros((mini_batch_size, self.N)).float()

    #     S[:, 0] = self.S_0
    #     I[:, 0] = 0

    #     for t in tqdm(range(self.N-1)):

    #         S[:, t+1], I[:,t+1], _ = self.step(t*self.dt, S[:,t], I[:,t], 0*I[:,t])

    #     return S, I
    
    def step(self, t, S, I, I_p):
        """
        evolves the sysmte from old state to new state and provides the reward.
        action = new inventory level

        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        S : TYPE
            DESCRIPTION.
        I : TYPE
            DESCRIPTION.
        I_p : TYPE
            DESCRIPTION.

        Returns
        -------
        S_p : TYPE
            DESCRIPTION.
        I_p : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.

        """
        
        
        mini_batch_size = S.shape[0]

        # dS_t = \kappa ( \theta - S_t) dt + \sigma dW_t        
        S_p = self.theta + (S-self.theta)*np.exp(-self.kappa*self.dt) \
            + self.eff_vol *  torch.randn(mini_batch_size)

        # quantity of assets that you must purchase to change from I to I_p
        q = I_p-I

        # reward recieve from the transaction
        #
        # X_t = \int_0^t (-S_u dI_u) + I_T S_T - \lambda \int_0^t  |dI_u|
        #
        # (follows from Ito's lemma and Ito product rule)
        # d( I_t S_t) = S_t dI_t + I_t dS_t + d[S,I]_t
        #             = S_t dI_t + I_t dS_t
        # I_T S_T  = \int_0^T S_u dI_u + \int_0^T I_u dS_u
        #
        # X_T =   \int_0^T I_u dS_u - \lambda \int_0^T  |dI_u|
        #     = \int_0^T ( I_u dS_u -  \lambda  |dI_u| )
        #
        r = I_p*(S_p-S) - self.lambd*torch.abs(q)
        

        return S_p, I_p, r
