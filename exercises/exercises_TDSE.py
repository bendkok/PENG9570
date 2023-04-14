# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 11:43:08 2023

@author: benda
"""

import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.linalg as sl
import scipy.integrate as si
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# import time

sns.set_theme(style="dark") # nice plots


def psi_single_initial(x, x0 = -20, sigmap = 0.2, p0 = 3, tau = 5):
    # Initial value for a wave function with one Gaussian wave
    # return np.sqrt( np.sqrt(2) * sigmap / (np.sqrt(np.pi)*(1-2j*sigmap**2*tau)) ) * np.exp( - (sigmap**2 * (x-x0)**2 / (1-2j*sigmap**2*tau)) + 1j*p0*x)
    
    term0 = np.sqrt(2/np.pi) * sigmap / (1 - 2j*sigmap**2 * tau)
    term1 = sigmap**2 * (x - x0)**2 / (1 - 2j * sigmap**2 * tau) 
    term2 = 1j * p0 * x
    psi0 = np.sqrt(term0) * np.exp(- term1 + term2)
    return psi0 


def psi_single_analytical(t, x, x0 = -20, sigmap = 0.2, p0 = 3, tau = 5):
    # Analytical solution for a wave function with one Gaussian wave
    N2 = np.sqrt(2/np.pi) * sigmap / np.sqrt(1+4*sigmap**4*(t-tau)**2)
    return N2 * np.exp( - 2 * sigmap**2 * (x - x0 - p0*t)**2 / (1 + 4*sigmap**4*(t-tau)**2) )


def psi_double_initial(x, x0 = -25, p0 = 3, sigmap0 = 0.2, tau0 = 5, x1 = 25, p1 = -3, sigmap1 = 0.2, tau1 = 5):
    # Initial value for a wave function with two Gaussian waves
    return (psi_single_initial(x, x0=x0, p0=p0, sigmap=sigmap0, tau=tau0) + psi_single_initial(x, x0=x1, p0=p1, sigmap=sigmap1, tau=tau1)) / np.sqrt(2)


def Crank_Nicolson(psi, F):
    # a numerical approximation
    psi_new = (np.identity(F.shape[0]) - F) @ psi
    psi_new = psi_new.T
    psi_new = np.linalg.inv(np.identity(F.shape[0]) + F) @ psi_new
    return np.ravel(psi_new)


def Magnus_propagator(psi, H_adjusted):
    # a better numerical approximation
    return H_adjusted @ psi


def make_3_point_Hamiltonian(n, h, dt, V=0):
    """
    The Hamiltonian when using 3-point finite difference to discretice the spatial derivative.
    """
    ones = np.ones(n)
    D2_3 = sp.diags( [ ones[1:], -2*ones, ones[1:]], [-1, 0, 1], format='coo') / (h*h) # second order derivative
    H_3 = - 1/2 * (D2_3 + V) # Hamiltonian
    exp_iH_3dt = sl.expm(-1j*H_3.todense()*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_3dt2 = .5j*dt*H_3                       # adjusted Hamiltonian to fit Crank Nicolson
    return exp_iH_3dt, iH_3dt2, H_3


def make_5_point_Hamiltonian(n, h, dt, V=0):
    """
    The Hamiltonian when using 5-point finite difference to discretice the spatial derivative.
    """
    ones = np.ones(n)
    D2_5 = sp.diags( [-ones[2:], 16*ones[1:], -30*ones, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*h*h) # second order derivative
    H_5 = - 1/2 * (D2_5 + V) # Hamiltonian
    exp_iH_5dt = sl.expm(-1j*H_5.todense()*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_5dt2 = .5j*dt*H_5                       # adjusted Hamiltonian to fit Crank Nicolson
    return exp_iH_5dt, iH_5dt2, H_5


def make_fft_Hamiltonian(n, L, dt, V=0):
    """
    The Hamiltonian when using Fourier transformation to discretice the spatial derivative.
    When we take the FFT, we can take the spatial derivative by simply multiplying by ik,
    and then transforming back.
    """
    k_fft = 2*(np.pi/L)*np.array(list(range(int(n/2))) + list(range(int(-n/2),0)))
    # Hfft  = 1/2 * sc.fft.ifft2(sp.diags(k_fft**2) @ sc.fft.fft2(np.diag(np.ones(n)))) + V # Hamiltonian
    Hfft  = 1/2 * sc.fft.ifft(sp.diags(k_fft**2) * sc.fft.fft(np.eye(n), axis=0), axis=0) + V # Hamiltonian 
    # Hfft  = 1/2 * sc.fft.ifft2(sp.diags(k_fft**2) @ sc.fft.fft(np.diag(np.ones(n)), axis=0)) + V # Hamiltonian
    # Hfft  = 1/2 * sc.fft.fft(sp.diags(k_fft**2) * sc.fft.fft(np.diag(np.ones(n)), axis=1),axis=1) + V # Hamiltonian
    # print(np.where(sc.fft.fft2(np.diag(np.ones(n)))> 1e-10)[0].shape, np.where(sc.fft.fft2(np.diag(np.ones(n)))> 1e-10)[1].shape )
    # print(np.where(sc.fft.fft(np.ones(n)) > 0))
    # Hfft  = 1/2 * sp.diags( sc.fft.ifft(k_fft**2 * sc.fft.fft(np.ones(n)) )) + V # Hamiltonian
    exp_iH_fft = sl.expm(-1j*Hfft*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_fftdt2  = .5j*dt*Hfft           # adjusted Hamiltonian to fit Crank Nicolson
    # print(exp_iH_fft.shape, iH_fftdt2.shape, Hfft.shape, k_fft.shape)
    # exit()
    return exp_iH_fft, iH_fftdt2, Hfft


def solve_while_plotting(x, psis0, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator, analytical=[], V=None, CAP=None):

    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(12, 8))
    # make the plots look a bit nicer
    ax.set_ylim(top = np.max(np.abs(psis0[0])**2)*2.2, bottom=-0.01)
    plt.xlabel(r"$x$")
    ax.set_ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    plt.grid()

    if V is not None or CAP is not None:
        ax_p = ax.twinx()
        align_yaxis(ax, ax_p, 1.3)
        ax_p.set_ylabel("Potential")

    if V is not None:
        line_V, = ax_p.plot(x, V.diagonal(), '--', color='tab:orange', label="Potential Barrier", zorder=2)
        # ax.set_ylim(top = np.max(np.abs(psis)**2)*1.5, bottom=-0.01)
        # align_yaxis(ax, ax_p, 1.3)
        # ax_p.set_ylabel("Potential")


    psis = psis0
    # plot the initial wave functions
    lines = [(ax.plot(x, np.abs(psis[i])**2, label=labels[i]))[0] for i in range(len(psis))]
    if len(analytical) > 0:
        line_anal, = ax.plot(x, analytical[0], '--', label="Analytical") # TODO: is behind the grid


    if len(psis0) == 1: # or psis0.count(psis0[0]) > 1:
        line_0, = ax.plot(x, np.abs(psis0[0])**2, 'g--', label=r"$\psi_0$", zorder=2)
        # plt.legend()
    elif (psis0[0] == psis0[1]).all():
        line_0, = ax.plot(x, np.abs(psis0[0])**2, 'g--', label=r"$\psi_0$", zorder=2)
    elif psis0.count(psis0[0]) == 1:
        lines0 = [(ax.plot(x, np.abs(psis0[i])**2, label=r"$\psi_0$ "+str(labels[i])))[0] for i in range(len(psis))]
        # plt.legend()

    # plt.legend()

    # ask matplotlib for the plotted objects and their labels
    if CAP is not None:
        CAP_vector, exp_CAP_vector_dt, CAP_locs = CAP
        line_CAP, = ax_p.plot(x, CAP_vector*np.max(V.diagonal())/np.max(CAP_vector), 'r--', label="CAP")
        # ax.set_ylim(top = np.max(np.abs(psis)**2)*1.5, bottom=-0.01)
        CAP_vector_r = np.zeros_like(CAP_vector)
        CAP_vector_l = np.zeros_like(CAP_vector)
        CAP_vector_r[CAP_locs[1]] = CAP_vector[CAP_locs[1]]
        CAP_vector_l[CAP_locs[2]] = CAP_vector[CAP_locs[2]]
        
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_p.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=1)
        # ax.set_ylim(top = np.max(psi_analytical(0, x))*1.1)

        ax.set_zorder(ax_p.get_zorder()+1) # put ax in front of ax_p
        ax.patch.set_visible(False)  # hide the 'canvas'
        ax_p.patch.set_visible(True) # show the 'canvas'
        
        Transmission = np.zeros(len(psis))
        Reflection   = np.zeros(len(psis))
        Reaminader   = np.zeros(len(psis))
        
        # dPl_dp = np.zeros((len(psis), len(CAP_locs[2])))
        # dPr_dp = np.zeros((len(psis), len(CAP_locs[1])))
        dPl_dp = np.zeros((len(psis), len(psis[0])))
        dPr_dp = np.zeros((len(psis), len(psis[0])))
        dP_dp  = np.zeros((len(psis), len(psis[0])))
        
        
    else:
        ax.legend()

    # goes through all the time steps
    for t in tqdm(range(len(times))):

        # finds the new values for psi
        for i in range(len(psis)):
            psis[i] = time_propagator(psis[i], Hamiltonians[i])
            
            if CAP is not None:
                overlap_R = np.trapz(CAP_vector[CAP_locs[1]] * np.abs(psis[i][CAP_locs[1]])**2, x[CAP_locs[1]])
                overlap_L = np.trapz(CAP_vector[CAP_locs[2]] * np.abs(psis[i][CAP_locs[2]])**2, x[CAP_locs[2]])
                
                # calculates the transmission and refflection this timestep
                Transmission[i] += overlap_R
                Reflection  [i] += overlap_L
                
                pis_fourier = np.conj( sc.fft.fft(psis[i]) )
                # dPr_dp[i][CAP_locs[1]] += np.real( pis_fourier[CAP_locs[1]] * sc.fft.fft(CAP_vector[CAP_locs[1]] * psis[i][CAP_locs[1]]) )
                # dPl_dp[i][CAP_locs[2]] += np.real( pis_fourier[CAP_locs[2]] * sc.fft.fft(CAP_vector[CAP_locs[2]] * psis[i][CAP_locs[2]]) )
                dPr_dp[i] += np.real( pis_fourier * sc.fft.fft(CAP_vector_r * psis[i]) )
                dPl_dp[i] += np.real( pis_fourier * sc.fft.fft(CAP_vector_l * psis[i]) )
                dP_dp [i] += np.real( pis_fourier * sc.fft.fft(CAP_vector   * psis[i]) )
                

        # we don't update the plot every single time step
        if t % plot_every == 0:
            # # finds the new values for psi
            # for i in range(len(psis)):
            #     # psis[i] = time_propagator(psis[i], Hamiltonians[i])
            #     psis[i] = time_propagator(psis0[i], Hamiltonians[i], (t)*dt)
            
            # if np.any(psis[i]-psis0[i] != 0.+0.j):
            #     print("afasfsad")

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
    
    if CAP is not None:
        
        dt = times[2]-times[1]
        
        Transmission = [Transmission[i] * 2 * dt for i in range(len(psis))]
        Reflection   = [Reflection  [i] * 2 * dt for i in range(len(psis))]
        Reaminader   = [np.sum(np.abs(psis[i])**2) for i in range(len(psis))]
    
        print()
        print(f"Transmission: {Transmission}.")
        print(f"Reflection:   {Reflection}.")
        print(f"Sum           {[Transmission[i] + Reflection[i] for i in range(len(psis))]}.")
        print(f"Reaminader:   {Reaminader}.", "\n")
        
        n = len(x)
        L = 2*np.max(x)
        dx = (np.max(x)-np.min(x))/(n-1)
        
        dPr_dp = dPr_dp * 2 * dt * dx**2 / (2*np.pi)
        dPl_dp = dPl_dp * 2 * dt * dx**2 / (2*np.pi)
        dP_dp  = dP_dp  * 2 * dt * dx**2 / (2*np.pi)
        
        phi2 = [np.fft.fftshift(i) for i in dP_dp]
        
        print()
        # print(f"dPr_dp: {dPr_dp}.")
        # print(f"dPl_dp: {dPl_dp}.")
        # print(f"Sum:    {[dPl_dp[i] + dPr_dp[i] for i in range(len(dPr_dp))]}.")
        # print(f"dP_dp:  {dP_dp}.", "\n")
        
        k_fft = 2*(np.pi/L)*np.array(list(range(int(n/2))) + list(range(int(-n/2),0)))
        k_fft = np.fft.fftshift(k_fft)
        # k = np.fft.fftshift( sc.fft.fftfreq(n, d=(x[2]-x[1])) ) # 2*(np.pi/L) *
        
        # check if it is properly normalised
        inte  = [np.trapz(phi2[p], k_fft) for p in range(len(psis))]
        print("Norm with manual k-vector: ", inte) # should be ~1
        
        peaks = sc.signal.find_peaks(phi2[0], height=0.9)
        print()
        print(f"Peak values: {phi2[0][peaks[0]]}.")
        print(f"Peak locs:   {k_fft[peaks[0]]}.", '\n') # " p0 = {p0}.", '\n')
        
        figure1, ax1 = plt.subplots(figsize=(12, 8))
        for p in range(len(psis)):
            ax1.plot(k_fft, np.fft.fftshift(dPr_dp)[p], '--', label="Right") # label=r"$dP_r/dt$")
            ax1.plot(k_fft, np.fft.fftshift(dPl_dp)[p], '--', label="Left") # label=r"$dP_l/dt$")
            ax1.plot(k_fft,                    phi2[p],       label="Total") # label=r"$dP/dp$" ) # /inte
        ax1.set_xlabel(r"$p$")
        # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
        ax1.set_ylabel(r"$dP/dp$") # TODO: find better name
        # plt.yscale("log")
        ax1.grid()
        ax1.legend()
        # title = "Double potential" if pot_2 == 1 else "Single potential"
        title = "Velocity density function with CAP." # +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
        plt.title(title)
        plt.show()
    
    return psis


def align_yaxis(ax1, ax2, scale_1=1, scale_2=1):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])
    y_lims[0,1] *= scale_1
    y_lims[1,1] *= scale_2

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)



def exe_1_5(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 100,
            T           = 20,
            plot_every  = 1,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau),psi_single_initial(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_3_point_Hamiltonian(n, h, dt)[1], make_5_point_Hamiltonian(n, h, dt)[1]]
    labels       = ["3-points", "5-points"]
    analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Crank_Nicolson, analytical=analytical)


def exe_1_6(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 200,
            T           = 35,
            plot_every  = 2,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau),psi_single_initial(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_3_point_Hamiltonian(n, h, dt)[0], make_5_point_Hamiltonian(n, h, dt)[0]]
    labels       = ["3-points", "5-points"]
    analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator, analytical=analytical)


def exe_1_7(x0          = -20,
            sigmap      = 0.2,
            p0          = 3,
            tau         = 5,
            L           = 100,
            n           = 1024,
            t_steps     = 200,
            T           = 35,
            plot_every  = 2,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau)]*3
    Hamiltonians = [make_3_point_Hamiltonian(n, h, dt)[0], make_5_point_Hamiltonian(n, h, dt)[0], make_fft_Hamiltonian(n, L, dt)[0]]
    labels       = ["3-points", "5-points", "FFT"]
    analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator, analytical=analytical)


def exe_1_8(x0          = -20,
            x1          =  20,
            sigmap      = 0.2,
            p0          = 3,
            p1          = -3,
            tau         = 5,
            L           = 100,
            n           = 512,
            t_steps     = 200,
            T           = 35,
            plot_every  = 2,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    # h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    psis         = [psi_double_initial(x, x0, p0, sigmap, tau, x1, p1, sigmap, tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, L, dt)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator)


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
            plot_every  = 3,
            V0          = 3,
            w           = 2,
            s           = 5,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    # h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    potential = rectangular_potential(x, V0, s, w)

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, L,dt, V=potential)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    res_psi = solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator, V=potential)

    trans_loc  = np.where(x>0)[0]
    trans_prob = np.abs(res_psi[0][trans_loc])**2
    trans_pro  = np.trapz(trans_prob, x[trans_loc])

    refle_loc  = np.where(x<=0)[0]
    refle_prob = np.abs(res_psi[0][refle_loc])**2
    refle_pro  = np.trapz(refle_prob, x[refle_loc])

    print(f"Transmission probability: {trans_pro}.")
    print(f"Reflection probability:   {refle_pro}.")
    print(f"Sum probability:          {trans_pro+refle_pro}.")

    # makes the plot window stay up until it is closed
    plt.ioff()
    plt.show()


def exe_2_3(x0          = -50,
            sigmap      = 0.2,
            p0          = 1,
            tau         = 0,
            L           = 300,
            n           = 512,
            t_steps     = 300,
            T           = 100,
            plot_every  = 3,
            V0          = 1,
            w           = 1,
            s           = 5,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    # h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    potential = rectangular_potential(x, V0, s, w)

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, L,dt, V=potential)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    res_psi = solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator, V=potential)

    trans_loc  = np.where(x>0)[0]
    trans_prob = np.abs(res_psi[0][trans_loc])**2
    trans_pro  = np.trapz(trans_prob, x[trans_loc])

    refle_loc  = np.where(x<=0)[0]
    refle_prob = np.abs(res_psi[0][refle_loc])**2
    refle_pro  = np.trapz(refle_prob, x[refle_loc])

    print(f"Transmission probability: {trans_pro}.")
    print(f"Reflection probability:   {refle_pro}.")
    print(f"Sum probability:          {trans_pro+refle_pro}.")

    # makes the plot window stay up until it is closed
    plt.ioff()
    plt.show()


def solve_no_plotting(psis, Hamiltonians):

    # finds the new values for psi
    psis_new = [0] * len(psis)
    for i in range(len(psis)):
        psis_new[i] = Magnus_propagator(psis[i], Hamiltonians[i])

    return psis_new


def exe_2_4(x0          = -60,
            sigmap      = 0.1,
            p0_min      = .3,
            p0_max      = 6,
            n_p0        = 200,
            tau         = 0,
            L           = 1000,
            n           = 4096,
            V0          = 2,
            w           = .5,
            s           = 25,
            d           = 2,
            pot_2       = 1,
            animate     = False,
            do_save     = False,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    # h = (np.max(x)-np.min(x))/n # physical step length

    potential = rectangular_potential(x+d, V0, s, w) + pot_2*rectangular_potential(x-d, V0, s, w)
    print(f"Max potential = {np.max(potential.diagonal())} of {V0}.")

    p0s = np.linspace(p0_min, p0_max, n_p0)

    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])


    trans_proability = np.zeros(len(p0s))
    trap_proability  = np.zeros(len(p0s))
    refle_proability = np.zeros(len(p0s))

    trans_loc  = np.where(x>d)[0]
    trap_loc   = np.where(np.abs(x)<d)[0]
    refle_loc  = np.where(x<=-d)[0]
    
    n = len(x)
    L = 2*np.max(x)
    dx = (np.max(x)-np.min(x))/(n-1)
    # k_fft = 2*(np.pi/L)*np.array(list(range(int(n/2))) + list(range(int(-n/2),0)))
    # k_fft = np.fft.fftshift(k_fft)
    k_fft = 2*(np.pi/L)*np.array(list(range(int(-n/2), int(n/2))))

    phi2s = np.zeros((len(p0s), len(x)))
    norms = np.zeros(len(p0s))
    
    for p in tqdm(range(len(p0s))):

        p0 = p0s[p]
        psi         = [psi_single_initial(x,x0,sigmap,p0,tau)]
        Hamiltonian = [make_fft_Hamiltonian(n, L,(L/4 - x0)/(p0), V=potential)[0]] # T = (L/4 - x0)/p0

        res_psi = solve_no_plotting(psi, Hamiltonian)[0]

        trans_prob = np.abs(res_psi[trans_loc])**2
        trans_proability[p] = np.trapz(trans_prob, x[trans_loc])

        trap_prob  = np.abs(res_psi[trap_loc])**2
        trap_proability [p] =   np.trapz(trap_prob, x[trap_loc])

        refle_prob = np.abs(res_psi[refle_loc])**2
        refle_proability[p] = np.trapz(refle_prob, x[refle_loc]) 
        
        phi2s[p] = np.fft.fftshift(np.abs(sc.fft.fft(res_psi))**2) 
        # norms[p] = si.simpson(phi2s[p], k_fft) 
        
    phi2s = phi2s * dx**2 / (2*np.pi)
    for p in range(len(p0s)):
        norms[p] = si.simpson(phi2s[p], k_fft) 
    
    # print(f"Transmission probability: {trans_pro}.")
    # print(f"Reflection probability:   {refle_pro}.")
    # print(f"Trapped probability:      {trap_pro}.")
    # print(f"Sum probability:          {trans_pro+refle_pro+trap_pro}.")

    # figure, ax = plt.subplots(figsize=(10, 8))
    # ax.set_ylim(top = np.max(np.abs(psis)**2)*2.2, bottom=-0.01)
    # line_trans, = ax.plot(p0s, trans_proability, label="Transmission")
    # line_refle, = ax.plot(p0s, refle_proability, label="Reflection")
    # line_trap,  = ax.plot(p0s, trap_proability,  label="Trapped")
    plt.plot(p0s, trans_proability, label="Transmission")
    plt.plot(p0s, refle_proability, label="Reflection")
    plt.plot(p0s, trap_proability,  label="Trapped")
    plt.xlabel(r"$p_0$")
    # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    plt.ylabel("Probaility")
    plt.grid()
    plt.legend()
    title = "double potential" if pot_2 == 1 else "single potential"
    # title = "2.4: " + title +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    title = "Transmission/reflection probability for " + title + " without CAP.\n" + r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    plt.title(title)
    if do_save:
        savename = "TR_results/TR_" + ("double" if pot_2 else "single") + "_noCAP.pdf"
        plt.savefig(savename)
    plt.show()
    
    max_diff = np.max(np.abs(np.array(trans_proability) + np.array(refle_proability) - 1))
    print(f"Max differnce of sums from 1: {max_diff}")
    
    
    # check if it is properly normalised
    print(f"Max norm: {np.max(norms)}. Min norm: {np.min(norms)}.") # should be ~1
    X,Y = np.meshgrid(p0s, k_fft)
    plt.contourf(X,Y, phi2s.T, norm="log")
    plt.xlabel(r"$p_0$")
    plt.ylabel(r"$k$")
    plt.colorbar(label=r"$dP/dp$")
    title = "double potential" if pot_2 == 1 else "single potential"
    title = "Velocity density distribution for " + title + " without CAP.\n" +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    plt.title(title)
    plt.ylim(-6.1,6.1)
    if do_save:
        savename = "dPdp_results/dP_dt_" + ("double" if pot_2 else "single") + "_noCAP.pdf"
        plt.savefig(savename)
    plt.show()
    

    # plt.plot(x, np.abs(res_psi)**2)
    # plt.plot(x, potential.diagonal(), '--') # /np.max(np.abs(res_psi)**2)
    # plt.ylim(top = np.max(np.abs(res_psi)**2) * 1.2, bottom = -0.01)
    # plt.show()

    # loc = np.where(np.abs(x) < 3*d)[0]
    # plt.plot(x[loc], potential.diagonal()[loc], 'o--')
    # plt.show()

    if animate:
        n2 = int(n_p0/2)
        exe_2_4_anim(x0,sigmap,p0s[ 0],tau,L,n,1000,(L/4 - x0)/(p0s[ 0]),1,V0,w,s,d)
        exe_2_4_anim(x0,sigmap,p0s[n2],tau,L,n,1000,(L/4 - x0)/(p0s[n2]),1,V0,w,s,d)
        exe_2_4_anim(x0,sigmap,p0s[-1],tau,L,n,1000,(L/4 - x0)/(p0s[-1]),1,V0,w,s,d)


def exe_2_4_anim(x0          = -50,
                 sigmap      = 0.1,
                 p0          = 1.8,
                 tau         = 0,
                 L           = 700,
                 n           = 2048,
                 t_steps     = 1000,
                 T0          = 1000,
                 plot_every  = 8,
                 V0          = 2,
                 w           = 1,
                 s           = 25,
                 d           = 2,
                 pot_2       = 1,
                 ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    # h = (np.max(x)-np.min(x))/n # physical step length
    
    T = (L/4 - x0)/(p0)
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    potential = rectangular_potential(x-d, V0, s, w) + pot_2*rectangular_potential(x+d, V0, s, w)
    
    print(f"Max potential = {np.max(potential.diagonal())} of {V0}.")

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, L,dt, V=potential)[0]]
    labels       = ["FFT"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    res_psi = solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, time_propagator=Magnus_propagator, V=potential)

    trans_loc  = np.where(x>d)[0]
    trans_prob = np.abs(res_psi[0][trans_loc])**2
    trans_pro  = np.trapz(trans_prob, x[trans_loc])

    trap_loc   = np.where(np.abs(x)<d)[0]
    trap_prob  = np.abs(res_psi[0][trap_loc])**2
    trap_pro   = np.trapz(trap_prob, x[trap_loc])

    refle_loc  = np.where(x<=-d)[0]
    refle_prob = np.abs(res_psi[0][refle_loc])**2
    refle_pro  = np.trapz(refle_prob, x[refle_loc])

    print(f"Transmission probability: {trans_pro}.")
    print(f"Reflection probability:   {refle_pro}.")
    print(f"Trapped probability:      {trap_pro}.")
    print(f"Sum probability:          {trans_pro+refle_pro+trap_pro}.")

    # makes the plot window stay up until it is closed
    plt.ioff()
    plt.show()
    
    n = len(x)
    L = 2*np.max(x)
    dx = (np.max(x)-np.min(x))/(n-1)
    phi2 = np.fft.fftshift(np.abs(sc.fft.fft(res_psi[0]))**2) * dx**2 / (2*np.pi)
        
    k_fft = 2*(np.pi/L)*np.array(list(range(int(n/2))) + list(range(int(-n/2),0)))
    k_fft = np.fft.fftshift(k_fft)
    
    # check if it is properly normalised
    inte  = si.simpson(phi2, k_fft) 
    print("Norm with manual k-vector: ", inte) # should be ~1
    
    # we find the peaks values
    peaks = sc.signal.find_peaks(phi2, height=np.max(phi2)*0.05)
    print()
    print(f"Peak values: {phi2[peaks[0]]}.")
    print(f"Peak locs:   {k_fft[peaks[0]]}. p0 = {p0}.", '\n')
    
    figure1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.plot(k_fft, phi2, label="No CAP") # label=r"$dP/dp$" )
    ax1.set_xlabel(r"$p$")
    # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    ax1.set_ylabel(r"$dP/dp$") # TODO: find better name
    ax1.set_xlim((-np.abs(p0)*2,np.abs(p0)*2))
    # plt.yscale("log")
    ax1.grid()
    ax1.legend()
    # title = "Double potential" if pot_2 == 1 else "Single potential"
    title = title = "Velocity density function without CAP." # title + " with CAP." +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    plt.title(title)
    plt.show()
    


def square_gamma_CAP(x, dt=1, gamma_0=1, R=160):

    CAP_locs = np.where(np.abs(x) > R)[0] # TODO: make more efficient
    CAP_R_locs = np.where(x >  R)[0]
    CAP_L_locs = np.where(x < -R)[0]
    Gamma_vector           = np.zeros_like(x)
    Gamma_vector[CAP_locs] = gamma_0*(np.abs(x[CAP_locs]) - R)**2  # if abs(x)>R else 0
    # Gamma_vector[CAP_locs] = gamma_0*(x[CAP_locs] - R)**2  # if abs(x)>R else 0
    exp_Gamma_vector_dt  = np.exp(-Gamma_vector*dt  )[:,None]  # when actually using Γ we are using one of these formulas
    # exp_Gamma_vector_dt2 = np.exp(-Gamma_vector*dt*2)[:,None]  # so we just calculate them here to save flops
    return Gamma_vector, exp_Gamma_vector_dt, [CAP_locs, CAP_R_locs, CAP_L_locs]


def exe_CAP(x0          = -30,
            sigmap      = 0.1,
            p0_min      = .2,
            p0_max      = 6,
            n_p0        = 200,
            tau         = 0,
            L           = 200,
            n           = 512,
            t_steps     = 200,
            V0          = 3,
            w           = .5,
            s           = 25,
            d           = 2,
            pot_2       = 1,
            animate     = False,
            gamma_0     = .005,
            R_part      = .8,
            do_save     = False,
            ):

    x = np.linspace(-L/2, L/2, n) # physical grid
    h = (np.max(x)-np.min(x))/(n-1) # physical step length

    potential = rectangular_potential(x+d, V0, s, w) + pot_2*rectangular_potential(x-d, V0, s, w)
    print(f"Max potential = {np.max(potential.diagonal())} of {V0}.")

    p0s = np.linspace(p0_min, p0_max, n_p0)
    gamma_0s = p0s * 3 / 2000

    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    # exp_iH_fft = make_fft_Hamiltonian(n, L,1, V=potential)[2]
    trans_proability = []
    trap_proability  = []
    refle_proability = []

    Transmission = np.zeros(len(p0s))
    Reflection   = np.zeros(len(p0s))
    Reaminader   = np.zeros(len(p0s))
    dPl_dp       = np.zeros((len(p0s), len(x)))
    dPr_dp       = np.zeros((len(p0s), len(x)))
    dP_dp        = np.zeros((len(p0s), len(x)))

    fininsh_l = []

    Ts  = (L/4 + np.abs(x0))/np.abs(p0s)
    dts = Ts/t_steps
    
    n = len(x)
    L = 2*np.max(x)
    dx = (np.max(x)-np.min(x))/(n-1)

    for p in tqdm(range(len(p0s))):

        p0  = p0s[p]
        # T   = Ts [p] # (L/4 + np.abs(x0))/np.abs(p0) # (L/4 - x0)/(p0)*4
        dt  = dts[p] # T/t_steps
        dt2 = dt*2
        # times       = np.linspace(dt, T, t_steps)
        # dt          = T[2] - T[1]
        psi         = [psi_single_initial(x,x0,sigmap,p0,tau)]
        CAP_vector, exp_CAP_vector_dt, CAP_locs = square_gamma_CAP(x, dt=dt, gamma_0=gamma_0s[p], R = R_part*L/2)
        Hamiltonian = [make_fft_Hamiltonian(n,L,dt, V = potential - sp.diags(1j*CAP_vector))[0]] # * exp_CAP_vector_dt] # [exp_iH_fft**dt * CAP[1]] # T = (L/4 - x0)/p0

        CAP_vector_r = np.zeros_like(CAP_vector)
        CAP_vector_l = np.zeros_like(CAP_vector)
        CAP_vector_r[CAP_locs[1]] = CAP_vector[CAP_locs[1]]
        CAP_vector_l[CAP_locs[2]] = CAP_vector[CAP_locs[2]]
        
        
        res_psi     = psi
        not_converged = True

        stop_test = 200 # int(300/p0)

        l = 0
        while not_converged: # np.sum(np.abs(res_psi)**2) > 1e-6:
            # for t in times:
            # Hamiltonian = [(exp_iH_fft * sl.expm(dt)) * CAP[1]] # T = (L/4 - x0)/p0
            res_psi = solve_no_plotting(res_psi, Hamiltonian)
            # print(np.max(res_psi[0]))

            # we find the part of the AF which overlaps with the right and left CAPs
            # if len(res_psi) > 1:
            #     print("")
            # if CAP_vector.shape[0] < np.max(CAP_locs[1]):
            #     print("")
            # if res_psi[0].shape < np.max(CAP_locs[1]):
            #     print("")
            overlap_R = np.trapz(CAP_vector[CAP_locs[1]] * np.abs(res_psi[0][CAP_locs[1]])**2, x[CAP_locs[1]])
            overlap_L = np.trapz(CAP_vector[CAP_locs[2]] * np.abs(res_psi[0][CAP_locs[2]])**2, x[CAP_locs[2]])
            
            # calculates the transmission and refflection this timestep
            Transmission[p] += overlap_R
            Reflection  [p] += overlap_L
            
            pis_fourier = np.conj( sc.fft.fft(res_psi[0]) )
            dPr_dp[p] += np.real(pis_fourier * sc.fft.fft(CAP_vector_r * res_psi[0]) ) 
            dPl_dp[p] += np.real(pis_fourier * sc.fft.fft(CAP_vector_l * res_psi[0]) )
            dP_dp[p]  += np.real(pis_fourier * sc.fft.fft(CAP_vector   * res_psi[0]) )
            # dPr_dp[p] = dPr_dp[p] + np.real( sc.fft.fft(CAP_vector_r * res_psi[0]) ) * pis_fourier 
            # dPl_dp[p] = dPl_dp[p] + np.real( sc.fft.fft(CAP_vector_l * res_psi[0]) ) * pis_fourier 
            # dP_dp[p]  = dP_dp[p]  + np.real( sc.fft.fft(CAP_vector   * res_psi[0]) ) * pis_fourier 

            l+=1
            if l % stop_test == 0:
                # print(l)
                if np.sum(np.abs(res_psi[0])**2)*h < 1e-6:
                    fininsh_l.append(l)
                    not_converged = False
                    Reaminader[p] = np.sum(np.abs(res_psi[0])**2)

        
        Transmission[p] *= dt2
        Reflection  [p] *= dt2
        
        # exit()
        res_psi = res_psi[0]

        trans_loc  = np.where(x>d)[0]
        trans_prob = np.abs(res_psi[trans_loc])**2
        trans_proability.append( np.trapz(trans_prob, x[trans_loc]) )

        trap_loc   = np.where(np.abs(x)<d)[0]
        trap_prob  = np.abs(res_psi[trap_loc])**2
        trap_proability.append(  np.trapz(trap_prob, x[trap_loc]))

        refle_loc  = np.where(x<=-d)[0]
        refle_prob = np.abs(res_psi[refle_loc])**2
        refle_proability.append(  np.trapz(refle_prob, x[refle_loc]) )
        
        # dPr_dp[p] = dPr_dp[p] * 2 * dt * dx**2 / (2*np.pi)
        # dPl_dp[p] = dPl_dp[p] * 2 * dt * dx**2 / (2*np.pi)
        # dP_dp [p] = dP_dp [p] * 2 * dt * dx**2 / (2*np.pi)
        dPr_dp[p] *= dt
        dPl_dp[p] *= dt
        dP_dp [p] *= dt
        
    dPr_dp *= 2 * dx**2 / (2*np.pi)
    dPl_dp *= 2 * dx**2 / (2*np.pi)
    dP_dp  *= 2 * dx**2 / (2*np.pi)

    # print(f"Transmission probability: {trans_pro}.")
    # print(f"Reflection probability:   {refle_pro}.")
    # print(f"Trapped probability:      {trap_pro}.")
    # print(f"Sum probability:          {trans_pro+refle_pro+trap_pro}.")

    # figure, ax = plt.subplots(figsize=(10, 8))
    # ax.set_ylim(top = np.max(np.abs(psis)**2)*2.2, bottom=-0.01)
    # line_trans, = ax.plot(p0s, trans_proability, label="Transmission")
    # line_refle, = ax.plot(p0s, refle_proability, label="Reflection")
    # line_trap,  = ax.plot(p0s, trap_proability,  label="Trapped")

    # plt.plot(p0s, trans_proability, label="Transmission")
    # plt.plot(p0s, refle_proability, label="Reflection")
    # plt.plot(p0s, trap_proability,  label="Trapped")
    # plt.xlabel(r"$p_0$")
    # # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    # plt.ylabel("Probaility")
    # plt.grid()
    # plt.legend()
    # title = "Double potential." if pot_2 == 1 else "Single potential."
    # title = title +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    # plt.title(title)
    # plt.show()

    # plt.plot(x, np.abs(res_psi)**2)
    # plt.plot(x, potential.diagonal(), '--') # /np.max(np.abs(res_psi)**2)
    # plt.ylim(top = np.max(np.abs(res_psi)**2) * 1.2, bottom = -0.01)
    # plt.show()

    # loc = np.where(np.abs(x) < 3*d)[0]
    # plt.plot(x[loc], potential.diagonal()[loc], 'o--')
    # plt.show()
    
    sums = [Transmission[s]+Reflection[s] for s in range(len(Transmission))]

    plt.plot(p0s, Transmission, label="Transmission")
    plt.plot(p0s, Reflection, label="Reflection")
    plt.plot(p0s, sums,  label="Sum")
    plt.plot(p0s, Reaminader,  label="Reaminader")
    plt.xlabel(r"$p_0$")
    # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    plt.ylabel("Probaility") # TODO: find better name
    plt.grid()
    plt.legend()
    title = "double potential" if pot_2 == 1 else "single potential"
    title = "Transmission/reflection probability for " + title + " with CAP.\n" +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    plt.title(title)
    if do_save:
        savename = "TR_results/TR_" + ("double" if pot_2 else "single") + "_CAP.pdf"
        plt.savefig(savename)
    plt.show()
    
    
    
    # k_fft = 2*(np.pi/L)*np.array(list(range(int(n/2))) + list(range(int(-n/2),0)))
    # k_fft = np.fft.fftshift(k_fft)
    k_fft = 2*(np.pi/L)*np.array(list(range(int(-n/2), int(n/2))))
    
    phi2s = np.array([np.fft.fftshift(p) for p in dP_dp])
    
    # check if it is properly normalised
    # inte  = si.simpson(dP_dp, k_fft) 
    # print("Norm with manual k-vector: ", inte) # should be ~1
    
    # inte  = [np.trapz(phi2s[p], k_fft) for p in range(len(phi2s))]
    
    norms = np.zeros(len(p0s))
    for p in range(len(p0s)):
        # norms[p] = si.simpson(dP_dp[p], k_fft) 
        norms[p] = si.simpson(phi2s[p], k_fft) 
    
    # we find the peaks values
    # peaks = sc.signal.find_peaks(dP_dp, height=np.max(dP_dp)*0.05)
    # print()
    # print(f"Peak values: {dP_dp[peaks[0]]}.")
    # print(f"Peak locs:   {k_fft[peaks[0]]}. p0 = {p0}.", '\n')
    
    # check if it is properly normalised
    print(f"Max norm: {np.max(norms)}. Min norm: {np.min(norms)}. Mean norm: {np.mean(norms)}.") # should be ~1
    # print(f"Max norm: {np.max(inte)}. Min norm: {np.min(inte)}. Mean norm: {np.mean(inte)}.") # should be ~1
    X,Y = np.meshgrid(p0s, k_fft)
    # plt.contourf(X,Y, phi2s.T, norm="log")
    plt.contourf(X,Y, np.abs(phi2s.T), norm="log")
    plt.xlabel(r"$p_0$")
    plt.ylabel(r"$k$")
    plt.colorbar(label=r"$dP/dp$")
    title = "double potential" if pot_2 == 1 else "single potential"
    title = "Velocity density distribution for " + title + " with CAP.\n" +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    plt.title(title)
    # plt.ylim(-6.1,6.1)
    if do_save:
        savename = "dPdp_results/dP_dt_" + ("double" if pot_2 else "single") + "_CAP.pdf"
        plt.savefig(savename)
    plt.show()
    
    
    
    # plt.plot(p0s, dPr_dp, label="Right") # label=r"$dP_r/dt$")
    # plt.plot(p0s, dPl_dp, label="Left") # label=r"$dP_l/dt$")
    # plt.plot(p0s, dP_dp,  label="Total") # label=r"$dP/dp$" )
    # # plt.plot(p0s, dPl_dp+dPr_dp,  label="Sum")
    # plt.xlabel(r"$p_0$")
    # # plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    # plt.ylabel(r"$dP/dp$") # TODO: find better name
    # # plt.yscale("log")
    # plt.grid()
    # plt.legend()
    # title = "Double potential" if pot_2 == 1 else "Single potential"
    # title = title + " with CAP." +  r" $V_0$" + f"= {V0}, d = {d}, w = {w}."
    # plt.title(title)
    # plt.show()


    # print(f"Transmission: {np.min(Transmission), np.max(Transmission)}.")
    # print(f"Reflection:   {Reflection}.")
    # print(f"Sum {sums}.")
    # print()
    # print(np.min(sums), np.max(sums), np.mean(sums), np.std(sums))
    # print()
    
    max_diff = np.max(np.abs(np.array(sums) - 1))
    print(f"Max differnce of sums from 1: {max_diff}")
    # print("dPr_dp:", np.min(dPr_dp), np.max(dPr_dp), np.mean(dPr_dp), np.std(dPr_dp))
    # print("dPl_dp:", np.min(dPl_dp), np.max(dPl_dp), np.mean(dPl_dp), np.std(dPl_dp))

    # print()
    # print(fininsh_l)

    if animate:
        n2 = int(n_p0/2)
        exe_CAP_anim(x0,sigmap,p0s[ 0],tau,L,n,fininsh_l[ 0],fininsh_l[ 0]*dts[ 0],int(fininsh_l[ 0]/200),V0,w,s,d,gamma_0s[ 0],R_part)
        exe_CAP_anim(x0,sigmap,p0s[n2],tau,L,n,fininsh_l[n2],fininsh_l[n2]*dts[n2],int(fininsh_l[n2]/200),V0,w,s,d,gamma_0s[n2],R_part)
        exe_CAP_anim(x0,sigmap,p0s[-1],tau,L,n,fininsh_l[-1],fininsh_l[-1]*dts[-1],int(fininsh_l[-1]/200),V0,w,s,d,gamma_0s[-1],R_part)
        # exe_CAP_anim(x0,sigmap,p0s[ 0],tau,L,n,1000,T,1,2,V0,w,s,d,gamma_0,R_part)
        # exe_CAP_anim(x0,sigmap,p0s[n2],tau,L,n,1000,T,1,2,V0,w,s,d,gamma_0,R_part)
        # exe_CAP_anim(x0,sigmap,p0s[-1],tau,L,n,1000,T,1,2,V0,w,s,d,gamma_0,R_part)


def exe_CAP_anim(x0          = -50,
                 sigmap      = 0.1,
                 p0          = 1.7,
                 tau         = 0,
                 L           = 400,
                 n           = 1024,
                 t_steps     = 200,
                 T0          = 100,
                 plot_every  = 2,
                 V0          = 2,
                 w           = 1,
                 s           = 25,
                 d           = 2,
                 gamma_      = .0045,
                 R_part      = .8,
                 pot_2       = 1,
                 ):


    T = np.max(((L/4 + np.abs(x0))/np.abs(p0)*2.5, T0))
    # print((L/4 - x0)/(p0)*2, L/4, L/4-x0, (p0)*2)
    # exit()

    gamma_0 = p0 * 2 / 2000

    x = np.linspace(-L/2, L/2, n) # physical grid
    # h = (np.max(x)-np.min(x))/n # physical step length
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    # regular_potential = rectangular_potential(x-d, V0, s, w) + rectangular_potential(x+d, V0, s, w)
    CAP_vector, exp_CAP_vector_dt, CAP_locs = square_gamma_CAP(x, dt=dt, gamma_0=gamma_0, R = R_part*L/2) # [:,0] # * 1j
    potential =  rectangular_potential(x-d, V0, s, w) + pot_2*rectangular_potential(x+d, V0, s, w)

    psis         = [psi_single_initial(x,x0,sigmap,p0,tau)]
    Hamiltonians = [make_fft_Hamiltonian(n, L, dt, V = potential - sp.diags(1j*CAP_vector))[0] ] # * CAP[1]]
    labels       = [r"FFT $\psi$"]
    # analytical   = np.array([psi_single_analytical(t, x, x0,sigmap,p0,tau) for t in times])

    res_psi = solve_while_plotting(x, psis, Hamiltonians, times, plot_every, labels, 
                                   time_propagator=Magnus_propagator, V=potential, CAP=[CAP_vector, exp_CAP_vector_dt, CAP_locs]) 
    
    # if np.any(psis[0]-res_psi[0] != 0.+0.j):
    #     print("afasfsad")
        
    # print(psis[0]-res_psi[0])
    
    trans_loc  = np.where(x>d)[0]
    trans_prob = np.abs(res_psi[0][trans_loc])**2
    trans_pro  = np.trapz(trans_prob, x[trans_loc])

    trap_loc   = np.where(np.abs(x)<d)[0]
    trap_prob  = np.abs(res_psi[0][trap_loc])**2
    trap_pro   = np.trapz(trap_prob, x[trap_loc])

    refle_loc  = np.where(x<=-d)[0]
    refle_prob = np.abs(res_psi[0][refle_loc])**2
    refle_pro  = np.trapz(refle_prob, x[refle_loc])

    print(f"Transmission probability: {trans_pro}.")
    print(f"Reflection probability:   {refle_pro}.")
    print(f"Trapped probability:      {trap_pro}.")
    print(f"Sum probability:          {trans_pro+refle_pro+trap_pro}.")

    # makes the plot window stay up until it is closed
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    # print("Exercise 1.5:")
    # exe_1_5()
    # print("\nExercise 1.6:")
    # exe_1_6()
    # print("\nExercise 1.7:")
    # exe_1_7()
    # print("\nExercise 1.8:")
    # exe_1_8()

    # print("\nExercise 2.1:")
    # exe_2_1()
    # print("\nExercise 2.3:")
    # exe_2_3()
    # print("\nExercise 2.4:")
    # exe_2_4(pot_2=1, animate=True)
    # exe_2_4(pot_2=0, animate=True)
    
    exe_2_4_anim(pot_2=0)
    exe_CAP_anim(pot_2=0)
    exe_2_4_anim(pot_2=1)
    exe_CAP_anim(pot_2=1)

    # print("\nCAP single potential: ")
    # exe_CAP(animate=False, pot_2=0, n_p0=200, do_save=True) 
    # print("\nCAP double potential: ")
    # exe_CAP(animate=False, pot_2=1, n_p0=200, do_save=True) 
    
    # print("\nNo CAP single potential: ")
    # exe_2_4(x0          = -30,
    #         sigmap      = 0.1,
    #         p0_min      = .2,
    #         p0_max      = 6,
    #         n_p0        = 200,
    #         tau         = 0,
    #         L           = 500,
    #         n           = 1024,
    #         V0          = 3,
    #         w           = .5,
    #         s           = 25,
    #         d           = 2,
    #         pot_2       = 0,
    #         animate     = False, 
    #         do_save     = True,)
    # print("\nNo CAP double potential: ")
    # exe_2_4(x0          = -30,
    #         sigmap      = 0.1,
    #         p0_min      = .2,
    #         p0_max      = 6,
    #         n_p0        = 200,
    #         tau         = 0,
    #         L           = 500,
    #         n           = 1024,
    #         V0          = 3,
    #         w           = .5,
    #         s           = 25,
    #         d           = 2,
    #         pot_2       = 1,
    #         animate     = False,
    #         do_save     = True,)
    
    # n2 = int(100/2)
    # x0=-30; sigmap=0.1; tau=0; L=500; n=1024; V0=3; w=.5; s=25; d=2
    # p0s = np.linspace(.1, 6, 100)
    # exe_2_4_anim(x0,sigmap,p0s[ 0],tau,L,n,1000,(L/4 - x0)/(p0s[ 0]),8,V0,w,s,d,0)
    # exe_2_4_anim(x0,sigmap,p0s[n2],tau,L,n,1000,(L/4 - x0)/(p0s[n2]),8,V0,w,s,d,0)
    # exe_2_4_anim(x0,sigmap,p0s[-1],tau,L,n,1000,(L/4 - x0)/(p0s[-1]),8,V0,w,s,d,0)
    
