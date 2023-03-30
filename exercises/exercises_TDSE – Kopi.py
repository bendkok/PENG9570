# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:08 2023

@author: benda
"""

import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.linalg as sl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

sns.set_theme(style="dark") # nice plots


def psi_single_inital(x, x0 = -20, sigmap = 0.2, p0 = 3, tau = 5):
    # Initial value for a wave function with one Gaussian wave
    return np.sqrt( np.sqrt(2) * sigmap / (np.sqrt(np.pi)*(1-2j*sigmap**2*tau)) ) * np.exp( - (sigmap**2 * (x-x0)**2 / (1-2j*sigmap**2*tau)) + 1j*p0*x)

def psi_single_analytical(t, x, x0 = -20, sigmap = 0.2, p0 = 3, tau = 5,): 
    # Analytical solution for a wave function with one Gaussian wave
    N2 = np.sqrt(2/np.pi) * sigmap / np.sqrt(1+4*sigmap**4*(t-tau)**2)
    return  N2 * np.exp( - 2 * sigmap**2 * (x - x0 - p0*t)**2 / (1 + 4*sigmap**4*(t-tau)**2) )

def psi_double_inital(x, x0 = -25, p0 = 3, sigmap0 = 0.2, tau0 = 5, x1 = 25, p1 = -3, sigmap1 = 0.2, tau1 = 5):
    # Initial value for a wave function with two Gaussian waves
    return (psi_single_inital(x, x0=x0, p0=p0, sigmap=sigmap0, tau=tau0) + psi_single_inital(x, x0=x1, p0=p1, sigmap=sigmap1, tau=tau1)) / np.sqrt(2)

def Crank_Nicolson(psi, F, dt):
    # a numerical approximation
    psi_new = ((np.identity(F.shape[0]) - F) @ psi)
    psi_new = psi_new.T
    psi_new = np.linalg.inv(np.identity(F.shape[0]) + F) @ psi_new
    return np.ravel(psi_new)

def Magnus_propagator(psi, H_adjusted, dt):
    # a numerical appoximation
    return H_adjusted @ psi


def make_3_point_Hamiltonian(n, h, L, dt):
    """
    The Hamiltonian when using 3-point finite difference to discretice the spatial derivative.
    """
    ones = np.ones(n)
    D2_3 = sp.diags( [ ones[1:], -2*ones, ones[1:]] , [-1, 0, 1], format='coo') / (h*h) # second order derivative
    H_3 = - 1/2 * D2_3 # Hamiltonian 
    exp_iH_3dt = sl.expm(-1j*H_3.todense()*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_3dt2 = .5j*dt*H_3                       # adjusted Hamiltonian to fit Crank Nicolson
    return exp_iH_3dt, iH_3dt2, H_3

def make_5_point_Hamiltonian(n, h, L, dt):
    """
    The Hamiltonian when using 5-point finite difference to discretice the spatial derivative.
    """
    ones = np.ones(n)
    D2_5 = sp.diags( [-ones[2:], 16*ones[1:], -30*ones, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*h*h) # second order derivative
    H_5 = - 1/2 * D2_5 # Hamiltonian 
    exp_iH_5dt = sl.expm(-1j*H_5.todense()*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_5dt2 = .5j*dt*H_5                       # adjusted Hamiltonian to fit Crank Nicolson
    return exp_iH_5dt, iH_5dt2, H_5

def make_fft_Hamiltonian(n, h, L, dt, V=0):
    """
    The Hamiltonian when using Fourier transformation to discretice the spatial derivative.
    When we take the FFT, we can take the spatial derivative by simply multiplying by ik,
    and then transforming back. 
    """
    k_fft = 2*(np.pi/L)*np.array([i for i in range(int(n/2))]+[ j for j in range(int(-n/2),0)] ) 
    Hfft  = 1/2 * sc.fft.ifft2(sp.diags(k_fft**2) @ sc.fft.fft2(np.diag(np.ones(n)))) + V # Hamiltonian 
    exp_iH_fft = sl.expm(-1j*Hfft*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_fftdt2 = .5j*dt*Hfft           # adjusted Hamiltonian to fit Crank Nicolson
    return exp_iH_fft, iH_fftdt2, Hfft


def solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Magnus_propagator, analytical=[], V=None):
    
    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    # make the plots look a bit nicer
    ax.set_ylim(top = np.max(np.abs(psis)**2)*2.2, bottom=-0.01)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    plt.grid()
    
    plt.title("t = 0.")
    
    # plot the initial wave functions
    lines = [(ax.plot(x, np.abs(psis[i])**2, label=labels[i]))[0] for i in range(len(psis))]
    if len(analytical) > 0:
        line_anal, = ax.plot(x, analytical[0], '--', label="Analytical")
    
    if V is not None:
        line_V, = ax.plot(x, V.diagonal(), '--', label="Potenital Barrier")
        ax.set_ylim(top = np.max(np.abs(psis)**2)*2.2, bottom=-0.01)
        
    plt.legend()
    # ax.set_ylim(top = np.max(psi_analytical(0, x))*1.1)
    
    # goes thorugh all the time steps
    for t in tqdm(range(len(times))):
        
        # finds the new values for psi
        for i in range(len(psis)):
            psis[i] = time_propegator(psis[i], Hamiltonians[i], dt)

        # we don't update the plot every single time step        
        if t % plot_every == 0:
            [lines[i].set_ydata(np.abs(psis[i])**2) for i in range(len(psis))]
            if len(analytical) > 0:
                line_anal.set_ydata( analytical[t] )
            
            plt.title("t = {:.2f}.".format(times[t]))
            
            # drawing updated values
            figure.canvas.draw()
        
            # This will run the GUI event
            # loop until all UI events
            # currently waiting have been processed
            figure.canvas.flush_events()
    
    return psis

def solve_no_plotting(x, psis, Hamiltonians):
    
    # finds the new values for psi
    for i in range(len(psis)):
        psis[i] = Magnus_propagator(psis[i], Hamiltonians[i], 0)
    
    return psis


def exe_1_5(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 100,
            T           = 20,
            n_saved     = 10,
            plot_every  = 1,
            ):
    
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)
    
    psis         = [psi_single_inital(x,x0,sigmap,p0,tau),psi_single_inital(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_3_point_Hamiltonian(n, h, L, dt)[1], make_5_point_Hamiltonian(n, h, L, dt)[1]]
    labels       = ["3-points", "5-points"]
    analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Crank_Nicolson, analytical=analytical)

def exe_1_6(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 200,
            T           = 35,
            n_saved     = 10,
            plot_every  = 2,
            ):
    
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)
    
    psis         = [psi_single_inital(x,x0,sigmap,p0,tau),psi_single_inital(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_3_point_Hamiltonian(n, h, L, dt)[0], make_5_point_Hamiltonian(n, h, L, dt)[0]]
    labels       = ["3-points", "5-points"]
    analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Magnus_propagator, analytical=analytical)

def exe_1_7(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 200,
            T           = 35,
            n_saved     = 10,
            plot_every  = 2,
            ):
    
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)
    
    psis         = [psi_single_inital(x,x0,sigmap,p0,tau)]*3
    Hamiltonians = [make_3_point_Hamiltonian(n, h, L, dt)[0], make_5_point_Hamiltonian(n, h, L, dt)[0], make_fft_Hamiltonian(n, h, L, dt)[0]]
    labels       = ["3-points", "5-points", "FFT"]
    analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Magnus_propagator, analytical=analytical)


def exe_1_8(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 200,
            T           = 35,
            n_saved     = 10,
            plot_every  = 2,
            ):
    
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)
    
    psis         = [psi_double_inital(x)]
    Hamiltonians = [make_fft_Hamiltonian(n, h, L, dt)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Magnus_propagator)
    

def rectangular_potential(x, V0, s, w):
    return sp.diags( V0 / (1 + np.exp(s * (np.abs(x) - w/2))))

def exe_2_1(x0          = -50,
            sigmap      = 0.2,
            p0          = 1,
            tau         = 0,
            L           = 300,
            n           = 512,
            t_steps     = 300,
            T           = 100,
            n_saved     = 10,
            plot_every  = 3,
            V0          = 3,
            w           = 2,
            s           = 5,
            ):
    
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)
    
    potential = rectangular_potential(x, V0, s, w)
    
    psis         = [psi_single_inital(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, h, L, dt, V=potential)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    res_psii = solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Magnus_propagator, V=potential)
    
    trans_loc  = np.where(x>0)[0]
    trans_prob = np.abs(res_psii[0][trans_loc])**2
    trans_pro  = np.trapz(trans_prob, x[trans_loc])
    
    refle_loc  = np.where(x<=0)[0]
    refle_prob = np.abs(res_psii[0][refle_loc])**2
    refle_pro  = np.trapz(refle_prob, x[refle_loc])
    
    print(f"Transmission probability: {trans_pro}.")
    print(f"Reflection probability:   {refle_pro}.")
    print(f"Sum probability:          {trans_pro+refle_pro}.")
    
    # makes the plot window stay up until it is closed
    plt.ioff()
    plt.show()


def exe_2_3(x0          = -70,
            sigmap      = 0.1,
            p0          = 1,
            tau         = 0,
            L           = 400,
            n           = 1024,
            t_steps     = 400,
            T           = 100,
            n_saved     = 10,
            plot_every  = 4,
            V0          = 3,
            w           = .3,
            s           = 5,
        # x0          = -50,
        #     sigmap      = 0.2,
        #     p0          = 1,
        #     tau         = 0,
        #     L           = 300,
        #     n           = 512,
        #     t_steps     = 300,
        #     T           = 100,
        #     n_saved     = 10,
        #     plot_every  = 3,
        #     V0          = 1,
        #     w           = 1,
        #     s           = 5,
            ):
    
    T = (L/4 - x0)/p0
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)
    
    d=5
    # potential = rectangular_potential(x, V0, s, w)
    potential = rectangular_potential(x+d, V0, s, w) + rectangular_potential(x-d, V0, s, w)
    
    psis         = [psi_single_inital(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, h, L, dt, V=potential)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    res_psii = solve_while_plotting(x, dt, psis, Hamiltonians, times, plot_every, labels, time_propegator=Magnus_propagator, V=potential)
    
    trans_loc  = np.where(x>0)[0]
    trans_prob = np.abs(res_psii[0][trans_loc])**2
    trans_pro  = np.trapz(trans_prob, x[trans_loc])
    
    refle_loc  = np.where(x<=0)[0]
    refle_prob = np.abs(res_psii[0][refle_loc])**2
    refle_pro  = np.trapz(refle_prob, x[refle_loc])
    
    print(f"Transmission probability: {trans_pro}.")
    print(f"Reflection probability:   {refle_pro}.")
    print(f"Sum probability:          {trans_pro+refle_pro}.")
    
    # makes the plot window stay up until it is closed
    plt.ioff()
    plt.show()
    
        
def exe_2_4(x0          = -70,
            sigmap      = 0.1,
            p0_min      = .01,
            p0_max      = 5,
            n_p0        = 50,
            tau         = 0,
            L           = 200,
            n           = 512,
            T           = 60,
            V0          = 2,
            w           = .2,
            s           = 5,
            d           = 1,
            ):
    
    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    
    potential = rectangular_potential(x+d, V0, s, w) + rectangular_potential(x-d, V0, s, w)
    
    p0s = np.linspace(p0_min, p0_max, n_p0)
    
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])
    
    
    trans_proability = []
    trap_proability  = []
    refle_proability = []
    
    for p in tqdm(range(len(p0s))):
        
        p0 = p0s[p]
        psi         = [psi_single_inital(x,x0,sigmap,p0,tau)]
        Hamiltonian = [make_fft_Hamiltonian(n, h, L, (L/4 - x0)/(p0), V=potential)[0]] # T = (L/4 - x0)/p0
        
        res_psi = solve_no_plotting(x, psi, Hamiltonian)[0]
        
        trans_loc  = np.where(x>d)[0]
        trans_prob = np.abs(res_psi[trans_loc])**2
        trans_proability.append( np.trapz(trans_prob, x[trans_loc]) )
        
        trap_loc   = np.where(np.abs(x)<d)[0]
        trap_prob  = np.abs(res_psi[trap_loc])**2
        trap_proability.append(  np.trapz(trap_prob, x[trap_loc]))
        
        refle_loc  = np.where(x<=-d)[0]
        refle_prob = np.abs(res_psi[refle_loc])**2
        refle_proability.append(  np.trapz(refle_prob, x[refle_loc]) )
    
    # print(f"Transmission probability: {trans_pro}.")
    # print(f"Reflection probability:   {refle_pro}.")
    # print(f"Trapped probability:      {trap_pro}.")
    # print(f"Sum probability:          {trans_pro+refle_pro+trap_pro}.")
    
    figure, ax = plt.subplots(figsize=(10, 8))
    # ax.set_ylim(top = np.max(np.abs(psis)**2)*2.2, bottom=-0.01)
    line_trans, = ax.plot(p0s, trans_proability, label="Transmission")
    line_refle, = ax.plot(p0s, refle_proability, label="Reflection")
    line_trap,  = ax.plot(p0s, trap_proability,  label="Trapped")
    plt.xlabel(r"$p_0$")
    # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    plt.ylabel("Probaility")
    plt.grid()
    plt.legend()
    # plt.title("t = 0.")
    plt.show()
    
    plt.plot(x, np.abs(res_psi)**2)
    plt.show()
    

if __name__ == "__main__":
    # exe_1_5()
    # exe_1_6()
    # exe_1_7()
    # exe_1_8()
    # exe_2_1()
    exe_2_3()
    # exe_2_4()










