# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:22:25 2023

@author: bendikst
"""

import numpy as np
import matplotlib.pyplot as plt


dPdp = np.load("psi_results/dPdp_reg_400.npy")
CAP_vector = np.load("psi_results/CAP_vector.npy")
k_vector = np.load("psi_results/k_vector.npy")
x_vector = np.load("psi_results/x_vector.npy")

plt.plot(x_vector, dPdp[0])
plt.show()
