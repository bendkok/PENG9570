# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:06:02 2023

@author: bendikst
"""

import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.linalg as sl
import scipy.integrate as si
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from tqdm import tqdm


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


def make_fft_Hamiltonian(n, L, dt, V=0):
    """
    The Hamiltonian when using Fourier transformation to discretise the spatial derivative.
    When we take the FFT, we can take the spatial derivative by simply multiplying by ik,
    and then transforming back.
    """
    k_fft = 2*(np.pi/L)*np.array(list(range(int(n/2))) + list(range(int(-n/2),0)))
    Hfft  = 1/2 * sc.fft.ifft(sp.diags(k_fft**2) * sc.fft.fft(np.eye(n), axis=0), axis=0) + V # Hamiltonian 
    exp_iH_fft = sl.expm(-1j*Hfft*dt) # adjusted Hamiltonian to fit the Magnus propagator
    iH_fftdt2  = .5j*dt*Hfft           # adjusted Hamiltonian to fit Crank Nicolson
    return exp_iH_fft, iH_fftdt2, Hfft


def align_yaxis(ax1, ax2, scale_1=1, scale_2=1):
    # Helper function for alining the live plots
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
    

def rectangular_potential(x, V0, s, w):
    return sp.diags( V0 / (1 + np.exp(s * (np.abs(x) - w/2)))) # TODO: mak sure this always reaches V0


def square_gamma_CAP(x, dt=1, gamma_0=1, R=160):
    # a complex absorbing potential  with a square gamma function
    # Makes it so that when the wave function reaches R it starts being absorbed
    
    CAP_locs = np.where(np.abs(x) > R)[0] # TODO: make more efficient
    CAP_R_locs = np.where(x >  R)[0]
    CAP_L_locs = np.where(x < -R)[0]
    Gamma_vector           = np.zeros_like(x)
    Gamma_vector[CAP_locs] = gamma_0*(np.abs(x[CAP_locs]) - R)**2  # if abs(x)>R else 0
    return Gamma_vector, [CAP_locs, CAP_R_locs, CAP_L_locs]


def initalise_plot_figures(x, psis_initial, plot_labels, k_fft=None, analytical_plot=[], V_plot=None, final_time=1, CAP_vector=None, do_dP_dp_plot=True, do_CAP_plot=True, do_plot_initial=False, plot_title=""):
    # WIP
    
    n = len(x)
    L = 2*np.max(x)
    dx = (np.max(x)-np.min(x))/(n-1)
    if k_fft is None:
        k_fft = 2*(np.pi/L)*np.array(list(range(int(-n/2), int(n/2))))
    
    # axes = [None]*3
    # line = [None]*8
    
    plt.ion()
    
    sns.set_theme(style="dark") # nice plots

    # here we are creating sub plots
    if do_dP_dp_plot:
        # create subplots
        figure, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9.5), layout='constrained')
        figure.subplots_adjust(hspace=0.5)
        
        ax2.grid()
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$dP/dp$")
        
        # plot initial velocity density function            
        lines_dP_dp   = np.array([ax2.plot(k_fft, np.fft.fftshift(np.abs(sc.fft.fft(psis_initial[p]))**2) * dx**2 / (2*np.pi), label=plot_labels[p])[0] for p in range(len(psis_initial))])
        lines_dP_dp0, = ax2.plot(k_fft, np.fft.fftshift(np.abs(sc.fft.fft(psis_initial[0]))**2) * dx**2 / (2*np.pi), "--", label="initial", zorder=1) 
            
        ax2.legend()
        # axes[1] = ax2
        
    else:
        figure, ax = plt.subplots(figsize=(12, 8))
        ax2 = None
        
        lines_dP_dp  = None
        lines_dP_dp0 = None
        
    # make the plots look a bit nicer
    ax.set_ylim(top = np.max(np.abs(psis_initial[0])**2)*2.2, bottom=-0.01)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    ax.grid()

    # plot the initial wave functions
    lines_psi = np.array([(ax.plot(x, np.abs(psis_initial[p])**2, label=plot_labels[p]))[0] for p in range(len(psis_initial))])
    
    if len(analytical_plot) > 0:
        line_anal, = ax.plot(x, analytical_plot[0], '--', label="Analytical") # TODO: is behind the grid
    else:
        line_anal = None
    
    # if we want to allways show the initial wave function in the background
    if do_plot_initial:
        # if there is only one wave function, or their initial value are the same
        if len(psis_initial) == 1 or (psis_initial[0] == psis_initial[1]).all(): # or psis_initial.count(psis_initial[0]) > 1:
            line_initial, = ax.plot(x, np.abs(psis_initial[0])**2, 'g--', label=r"$\psi_0$", zorder=2)
        # if there are several distinct initial wave functions
        elif psis_initial.count(psis_initial[0]) == 1:
            line_initial = np.array([(ax.plot(x, np.abs(psis_initial[i])**2, label=r"$\psi_0$ "+str(plot_labels[i])))[0] for i in range(len(psis_initial))])
    else:
        line_initial = None
    
    # if we want an outline for V and/or the CAP
    if V_plot is not None or (do_CAP_plot and CAP_vector is not None):
        ax_p = ax.twinx()
        
        if V_plot is not None:
            line_V, = ax_p.plot(x, V_plot, '--', color='tab:orange', label="Potential Barrier", zorder=2)
        else:
            line_V = None
        
        # plot CAP outline, and add to legend
        if do_CAP_plot and CAP_vector is not None:
            line_CAP, = ax_p.plot(x, CAP_vector*np.max(V_plot)/np.max(CAP_vector), 'r--', label="CAP") # TODO: change scaling here
            
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_p.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc=1)
    
            ax.set_zorder(ax_p.get_zorder()+1) # put ax in front of ax_p
            ax.patch.set_visible(False)  # hide the 'canvas'
            ax_p.patch.set_visible(True) # show the 'canvas'
        else:
            line_CAP = None
        
        align_yaxis(ax, ax_p, 1.3)
        ax_p.set_ylabel("Potential")
        
        # axes[2] = ax_p
    
    # if not we just add a legend
    else:
        ax.legend()
        ax_p = None
        
        line_V   = None
        line_CAP = None
    
    figure.suptitle(plot_title + f" t = 0 of {final_time}.")
    
    # axes[0] = ax
    
    return figure, ax, ax2, ax_p, lines_psi, lines_dP_dp, lines_dP_dp0, line_anal, line_initial, line_V, line_CAP
    # return figure, axes, lines_dP_dp, lines_dP_dp0, lines_psi, line_anal, line_initial, line_V, line_CAP


def solve_once(x, psis_initial, Hamiltonians, times, time_propagator=Magnus_propagator, do_live_plot=True, do_dP_dp_plot=True, plot_every=1, plot_labels=[], analytical_plot=[], do_plot_initial=True, V_plot=None, CAP=None, do_save_psi=False, do_save_dPdp=False, save_every=1, do_CAP_plot=True, plot_title="Scattering.", midpoint=0):
    """
    Solves the TDSE for the given Ψs and Hamiltonians. The result can be plotted live if do_live_plot is True.
    """
    
    if len(times) % plot_every != 0:
        print("WARNING: The length of times should be divisible by plot_every.")        
    
    psis = psis_initial
    
    dt = times[2]-times[1]
    
    n = len(x)
    L = 2*np.max(x)
    dx = (np.max(x)-np.min(x))/(n-1)
    k_fft = 2*(np.pi/L)*np.array(list(range(int(-n/2), int(n/2))))
    
    dtdx2_pi = dt * dx**2 / np.pi
    dx2_2pi  = dx**2 / (2*np.pi)
    
    # if there is a CAP we can calculate a lot of values on the fly
    if CAP is not None:
        CAP_vector, CAP_locs = CAP
        CAP_vector_r = np.zeros_like(CAP_vector)
        CAP_vector_l = np.zeros_like(CAP_vector)
        CAP_vector_r[CAP_locs[1]] = CAP_vector[CAP_locs[1]]
        CAP_vector_l[CAP_locs[2]] = CAP_vector[CAP_locs[2]]
        
        Transmission = np.zeros(len(psis))
        Reflection   = np.zeros(len(psis))
        Remainder    = np.zeros(len(psis))
        
        # dPl_dp = np.zeros((len(psis), len(psis[0])))
        # dPr_dp = np.zeros((len(psis), len(psis[0])))
        dP_dp  = np.zeros((len(psis), len(psis[0])))
    
    
    if do_live_plot: # TODO: consider moving this to another function
    
        plt.ion()
        
        sns.set_theme(style="dark") # nice plots
    
        # here we are creating sub plots
        if do_dP_dp_plot:
            # create subplots
            figure, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9.5), layout='constrained')
            figure.subplots_adjust(hspace=0.5)
            
            ax2.grid()
            ax2.set_xlabel(r"$k$")
            ax2.set_ylabel(r"$dP/dp$")
            
            # plot initial velocity density function            
            lines_dP_dp   = np.array([ax2.plot(k_fft, np.fft.fftshift(np.abs(sc.fft.fft(psis_initial[p]))**2) * dx**2 / (2*np.pi), label=plot_labels[p])[0] for p in range(len(psis))])
            lines_dP_dp0, =  ax2.plot(k_fft, np.fft.fftshift(np.abs(sc.fft.fft(psis_initial[0]))**2) * dx**2 / (2*np.pi), "--", label="Initial", zorder=1) 
                
            ax2.legend()
            
        else:
            figure, ax = plt.subplots(figsize=(12, 8))
            
        # make the plots look a bit nicer
        ax.set_ylim(top = np.max(np.abs(psis_initial[0])**2)*2.2, bottom=-0.01)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
        ax.grid()
    
        # plot the initial wave functions
        lines_psi = np.array([(ax.plot(x, np.abs(psis[p])**2, label=plot_labels[p]))[0] for p in range(len(psis))])
        
        if len(analytical_plot) > 0:
            line_anal, = ax.plot(x, analytical_plot[0], '--', label="Analytical") # TODO: is behind the grid
        
        # if we want to allways show the initial wave function in the background
        if do_plot_initial:
            # if there is only one wave function, or their initial value are the same
            if len(psis_initial) == 1 or (psis_initial[0] == psis_initial[1]).all(): # or psis_initial.count(psis_initial[0]) > 1:
                line_initial, = ax.plot(x, np.abs(psis_initial[0])**2, 'g--', label=r"$\psi_0$", zorder=2)
            # if there are several distinct initial wave functions
            elif psis_initial.count(psis_initial[0]) == 1:
                line_initial = np.array([(ax.plot(x, np.abs(psis_initial[i])**2, label=r"$\psi_0$ "+str(plot_labels[i])))[0] for i in range(len(psis))])
        
        # if we want an outline for V and/or the CAP
        if V_plot is not None or (do_CAP_plot and CAP is not None):
            ax_p = ax.twinx()
            
            if V_plot is not None:
                line_V, = ax_p.plot(x, V_plot, '--', color='tab:orange', label="Potential Barrier", zorder=2)
            
            # plot CAP outline, and add to legend
            if do_CAP_plot and CAP is not None:
                line_CAP, = ax_p.plot(x, CAP_vector*np.max(V_plot)/np.max(CAP_vector), 'r--', label="CAP") # TODO: change scaling here
                
                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax_p.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc=1)
        
                ax.set_zorder(ax_p.get_zorder()+1) # put ax in front of ax_p
                ax.patch.set_visible(False)  # hide the 'canvas'
                ax_p.patch.set_visible(True) # show the 'canvas'
            
            align_yaxis(ax, ax_p, 1.3)
            ax_p.set_ylabel("Potential")
        
        # if not we just add a legend
        else:
            ax.legend()
        
        plt.title(plot_title + f" t = 0 of {times[-1]}.")
        
    
    """
    In the next part, to avoid doing many if-test during the loop, we seperate the functionality out into various functions,
    and then use if-test to see which we add to lists we go through. 
    There is one general list, and one for updating the plots.
    """
    
    def update_CAP():
        for p in range(len(psis)):
            # calculates the Transmission and reflection this timestep
            Transmission[p] += np.trapz(CAP_vector[CAP_locs[1]] * np.abs(psis[p][CAP_locs[1]])**2, x[CAP_locs[1]]) # overlap_R
            Reflection  [p] += np.trapz(CAP_vector[CAP_locs[2]] * np.abs(psis[p][CAP_locs[2]])**2, x[CAP_locs[2]]) # overlap_L
            
            # calculates the momentum distribution this timestep
            # dP_dp_change = np.real( np.conj( sc.fft.fft(psis[p])) * sc.fft.fft(CAP_vector   * psis[p]) ) 
            dP_dp [p] += np.real( np.conj( sc.fft.fft(psis[p] ) ) * sc.fft.fft( CAP_vector * psis[p] ) ) 
            
    def update_plots():
        # we don't update the plot every single time step
        if t % plot_every == 0:
            
            for p in range(len(psis)):
                abs_psi  = np.abs(psis[p])
                abs_psi2 = abs_psi**2
                lines_psi[p].set_ydata(abs_psi2) # update numeric wave functions
            
                for func in funcs_plot_list:
                    func(abs_psi, abs_psi2)
                
            plt.title(plot_title + " t = {:.2f} of {:.2f}.".format(times[t-1], times[-1]))

            # drawing updated values
            figure.canvas.draw()

            # This will run the GUI event
            # loop until all UI events
            # currently waiting have been processed
            figure.canvas.flush_events()
            
    def save_psis():
        # we don't save every single time step
        # if t % plot_every == 0:
        # for p in range(len(psis)):
        np.save(savename_psi.format(t), np.abs(psis[:,::save_every])**2)
            
    def save_dPdp():
        # we don't save every single time step
        # if t % plot_every == 0:
        # for p in range(len(psis)):
        np.save(savename_dPdp.format(t), np.fft.fftshift(np.abs(sc.fft.fft(psis[:,::save_every]))**2) * dx2_2pi) # , axis=0
            
    def save_dPdp_CAP():
        # we don't save every single time step
        # if t % plot_every == 0:
        # for p in range(len(psis)):
        np.save(savename_dPdp.format(t), np.abs(np.fft.fftshift(dP_dp[:,::save_every]) * dtdx2_pi)) # , axis=0
    
    def update_anal_plot(abs_psi=None, abs_psi2=None):
        line_anal.set_ydata( analytical_plot[t] ) # update analytical wave function
    
    p=0
    
    def update_dPdp_plot_CAP(abs_psi=None, abs_psi2=None):
        # for p in range(len(psis)):
        # lines_dP_dp[p].set_ydata(np.fft.fftshift(abs_psi * dtdx2_pi))
        lines_dP_dp[p].set_ydata(np.abs(np.fft.fftshift(dP_dp[p] * dtdx2_pi)))
    
    def update_dPdp_plot(abs_psi=None, abs_psi2=None):
        # for p in range(len(psis)):
        # lines_dP_dp[p].set_ydata(np.fft.fftshift(sc.fft.fft(abs_psi2)) * dx2_2pi)
        lines_dP_dp[p].set_ydata(np.fft.fftshift(np.abs(sc.fft.fft(psis[p]))**2) * dx2_2pi)
    
    funcs_list = []
    funcs_plot_list = []
    
    if CAP is not None:
        funcs_list.append(update_CAP)
    
    if do_live_plot:
        funcs_list.append(update_plots)
    
    if do_save_psi:
        # this gives a savename with an appropriate amount of 0s
        np.save("psi_results/x_vector", x[::save_every])
        np.save("psi_results/V_plot", V_plot[::save_every])
        np.save("psi_results/times", times)
        np.save("psi_results/psis_initial", psis_initial[:,::save_every])
        if CAP is not None:
            np.save("psi_results/CAP_vector", CAP_vector[::save_every])
        savename_psi = f"psi_results/psi_{'CAP' if CAP is not None else 'reg'}_" + "{:0"+str(int(np.log10(len(times)))+1)+"}" 
        funcs_list.append(save_psis)
        
    if do_save_dPdp:
        # this gives a savename with an appropiate amount of 0s
        np.save("psi_results/k_vector", k_fft[::save_every])
        savename_dPdp = f"psi_results/dPdp_{'CAP' if CAP is not None else 'reg'}_" + "{:0"+str(int(np.log10(len(times)))+1)+"}" # "{:05}"
        if CAP is not None:
            funcs_list.append(save_dPdp_CAP)
        else:
            funcs_list.append(save_dPdp)
    
    if len(analytical_plot) > 0: 
        funcs_plot_list.append(update_anal_plot)
    
    if do_dP_dp_plot: # and (np.abs(dP_dp_change[loc]) > 1e-1):
        # print(dP_dp_change, dtdx2_pi)
        if CAP is not None:
            funcs_plot_list.append(update_dPdp_plot_CAP)
        else:
            funcs_plot_list.append(update_dPdp_plot)
    
    
    # goes through all the time steps
    for t in tqdm(range(len(times))):
        
        for func in funcs_list:
            func()
            
        # finds the new values for psi
        for p in range(len(psis)):
            psis[p] = time_propagator(psis[p], Hamiltonians[p])
            
    # ensure that we also get the final timestep
    t = len(times)
    for func in funcs_list:
        func()  
        
        
            # update the CAP values
            # if CAP is not None:
            #     # calculates the Transmission and reflection this timestep
            #     Transmission[p] += np.trapz(CAP_vector[CAP_locs[1]] * np.abs(psis[p][CAP_locs[1]])**2, x[CAP_locs[1]]) # overlap_R
            #     Reflection  [p] += np.trapz(CAP_vector[CAP_locs[2]] * np.abs(psis[p][CAP_locs[2]])**2, x[CAP_locs[2]]) # overlap_L
                
            #     # calculates the momentum distribution this timestep
            #     dP_dp_change = np.real( np.conj( sc.fft.fft(psis[p])) * sc.fft.fft(CAP_vector   * psis[p]) ) 
            #     dP_dp [p] += dP_dp_change # np.real( np.conj( sc.fft.fft(psis[p])) * sc.fft.fft(CAP_vector   * psis[p]) ) 
                
        # # we don't update the plot every single time step
        # if do_live_plot and t % plot_every == 0:
            
        #     for func in funcs_plot_list:
        #         func()
            
        #     for p in range(len(psis)):
        #         lines_psi[p].set_ydata(np.abs(psis[p])**2) # update numeric wave functions
                
        #     # if len(analytical_plot) > 0: 
        #     #     line_anal.set_ydata( analytical_plot[t] ) # update analytical wave function
            
        #     # update momentum probability distribution
        #     # if do_dP_dp_plot: # and (np.abs(dP_dp_change[loc]) > 1e-1):
        #     #     # print(dP_dp_change, dtdx2_pi)
        #     #     if CAP is not None:
        #     #         # phi2 = [np.abs(np.fft.fftshift(i)) for i in (dP_dp * dtdx2_pi)]
        #     #         for p in range(len(psis)):
        #     #             # lines_dP_dp[p].set_ydata(phi2[p])
        #     #             lines_dP_dp[p].set_ydata(np.abs(np.fft.fftshift(dP_dp[p] * dtdx2_pi)))
        #     #     else:
        #     #         for p in range(len(psis)):
        #     #             lines_dP_dp[p].set_ydata(np.fft.fftshift(np.abs(sc.fft.fft(psis[p]))**2) * dx2_2pi)
            
        #     plt.title(plot_title + " t = {:.2f} of {:.2f}.".format(times[t], times[-1]))

        #     # drawing updated values
        #     figure.canvas.draw()

        #     # This will run the GUI event
        #     # loop until all UI events
        #     # currently waiting have been processed
        #     figure.canvas.flush_events()
    
    if do_live_plot:
        # makes the plot window stay up until it is closed
        plt.ioff()
        plt.show()
    
    
    if CAP is not None:
        Transmission = Transmission * 2 * dt # [Transmission[i] * 2 * dt for i in range(len(psis))]
        Reflection   = Reflection   * 2 * dt # [Reflection  [i] * 2 * dt for i in range(len(psis))]
        Remainder    = np.array([np.sum(np.abs(psis[i])**2) for i in range(len(psis))])
        
        phi2 = np.array([np.fft.fftshift(i) for i in (dP_dp * dtdx2_pi)])
        
        inte = si.simpson(phi2, k_fft) 
        
        return psis, Transmission, Reflection, Remainder, phi2, inte, x, k_fft
    
    else:    
        Transmission_loc  =  np.where(x>midpoint)[0]
        Transmission_prob = np.array([np.abs(psis[p][Transmission_loc])**2 for p in range(len(psis))])
        Transmission_pro  = np.array([np.trapz(Transmission_prob[p], x[Transmission_loc]) for p in range(len(psis))])
        
        Trapped_loc   =  np.where(np.abs(x)<midpoint)[0]
        Trapped_prob  = np.array([np.abs(psis[p][Trapped_loc])**2 for p in range(len(psis))])
        Trapped_pro   = np.array([np.trapz(Trapped_prob[p], x[Trapped_loc]) for p in range(len(psis))])
        
        Reflection_loc  =  np.where(x<=midpoint)[0]
        Reflection_prob = np.array([np.abs(psis[p][Reflection_loc])**2 for p in range(len(psis))])
        Reflection_pro  = np.array([np.trapz(Reflection_prob[p], x[Reflection_loc]) for p in range(len(psis))])
        
        phi2 = np.array([np.fft.fftshift(np.abs(sc.fft.fft(psis[p]))**2) * dx**2 / (2*np.pi) for p in range(len(psis))])
        
        inte  = si.simpson(phi2, k_fft) 
    
        return psis, Transmission_pro, Reflection_pro, Trapped_pro, phi2, inte, x, k_fft



def load_and_plot(load_folder="psi_results", use_CAP=True, do_CAP_plot=True, do_dP_dp_plot=True, plot_title="Scattering"):
    
    x = np.load(load_folder+"/x_vector.npy")
    k = np.load(load_folder+"/k_vector.npy")
    
    times = np.load(load_folder+"/times.npy")
    
    # initalise_plot_figures(x, psis_initial, plot_labels, k=k, V_plot=V_plot, final_time=times[-1], CAP_vector=CAP_vector, do_dP_dp_plot=do_dP_dp_plot, do_CAP_plot=use_CAP, do_plot_initial=True, plot_title="Scattering")
    dt = times[2]-times[1]
    n = len(x)
    L = 2*np.max(x)
    dx = (np.max(x)-np.min(x))/(n-1)
    if k is None:
        k = 2*(np.pi/L)*np.array(list(range(int(-n/2), int(n/2))))
    dtdx2_pi = dt * dx**2 / np.pi
    dx2_2pi  = dx**2 / (2*np.pi)
    
    load_psi  = load_folder+f"/psi_{'CAP' if use_CAP else 'reg'}_" + "{:0"+str(int(np.log10(len(times)))+1)+"}.npy" 
    psis_initial      = np.load(load_psi.format(0)) # abs of inital wave function squared
    psis_initial_flat = np.load(load_folder+"/psis_initial.npy") # inital wave function
    
    V_plot = np.load(load_folder+"/V_plot.npy")
    if do_CAP_plot and use_CAP:
        CAP_vector = np.load(load_folder+"/CAP_vector.npy")
    else:
        CAP_vector = None
        
    if do_dP_dp_plot:
        load_dPdp = load_folder+f"/dPdp_{'CAP' if use_CAP else 'reg'}_" + "{:0"+str(int(np.log10(len(times)))+1)+"}.npy"
        
        
        dPdp_initial = np.fft.fftshift(np.abs(sc.fft.fft(psis_initial_flat))**2) * dx2_2pi # np.load(load_dPdp.format(0)) # np.load(load_folder+"/dPdp_CAP_"+"0"*(int(np.log10(len(times)))+1)+".npy")
    
    # plot_labels = [r"$\psi$" for i in range(len(psis_initial))]
    plot_labels = ["Wave Function" for i in range(len(psis_initial))]
    
    
    # plt.ion()
    
    sns.set_theme(style="dark") # nice plots

    # here we are creating sub plots
    if do_dP_dp_plot:
        # create subplots
        figure, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 9.5), layout='constrained')
        # figure.subplots_adjust(hspace=0.5)
        
        ax2.grid()
        ax2.set_xlabel(r"$k$")
        ax2.set_ylabel(r"$dP/dp$")
        
        # plot initial velocity density function            
        lines_dP_dp   = [ax2.plot(k, dPdp_initial[p], label=plot_labels[p])[0] for p in range(len(psis_initial))]
        lines_dP_dp0, =  ax2.plot(k, dPdp_initial[0], "--", label="Initial", zorder=1) 
            
        ax2.legend()
        
    else:
        figure, ax = plt.subplots(figsize=(12, 8))
        
    # make the plots look a bit nicer
    ax.set_ylim(top = np.max(psis_initial[0])*2., bottom=-0.01)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    ax.grid()

    # plot the initial wave functions
    lines_psi = [(ax.plot(x, psis_initial[p], label=plot_labels[p]))[0] for p in range(len(psis_initial))]
    
    
    # we allways show the initial wave function in the background
    # if there is only one wave function, or their initial value are the same
    if len(psis_initial) == 1 or (psis_initial[0] == psis_initial[1]).all(): # or psis_initial.count(psis_initial[0]) > 1:
        # line_initial, = ax.plot(x, np.abs(psis_initial[0])**2, 'g--', label=r"$\psi_0$", zorder=2)
        line_initial, = ax.plot(x, psis_initial[0], 'g--', label=r"$\psi_0$", zorder=2)
    # if there are several distinct initial wave functions
    elif psis_initial.count(psis_initial[0]) == 1:
        # line_initial = np.array([(ax.plot(x, np.abs(psis_initial[i])**2, label=r"$\psi_0$ "+str(plot_labels[i])))[0] for i in range(len(psis_initial))])
        line_initial = np.array([(ax.plot(x, psis_initial[i], label=r"Initial $\psi$ "+str(plot_labels[i])))[0] for i in range(len(psis_initial))])
    
    # if we want an outline for V and/or the CAP
    if V_plot is not None or (do_CAP_plot and use_CAP):
        ax_p = ax.twinx()
        
        if V_plot is not None:
            line_V, = ax_p.plot(x, V_plot, '--', color='tab:orange', label="Potential Barrier", zorder=2)
        
        # plot CAP outline, and add to legend
        if do_CAP_plot and use_CAP:
            line_CAP, = ax_p.plot(x, CAP_vector*np.max(V_plot)/np.max(CAP_vector), 'r--', label="CAP") # TODO: change scaling here
            
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_p.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc=1)
    
            ax.set_zorder(ax_p.get_zorder()+1) # put ax in front of ax_p
            ax.patch.set_visible(False)  # hide the 'canvas'
            ax_p.patch.set_visible(True) # show the 'canvas'
        else:
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax_p.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc=1)
    
            ax.set_zorder(ax_p.get_zorder()+1) # put ax in front of ax_p
            ax.patch.set_visible(False)  # hide the 'canvas'
            ax_p.patch.set_visible(True) # show the 'canvas
        
        align_yaxis(ax, ax_p, 1.3)
        # ax_p.set_ylabel("Potential")
    
    # if not we just add a legend
    else:
        ax.legend()
    
    time_text = ax.text(.07, 0.95, f" t = 0 of {times[-1]}.", bbox=dict(facecolor='none', edgecolor='red'), horizontalalignment='left',verticalalignment='top', transform=ax.transAxes) # fig.suptitle("t = {:d} of {:d}.".format(0, 200)) 
    plt.title(plot_title) # + f" t = 0 of {times[-1]}.")
    if do_dP_dp_plot:
        lines_psi = lines_psi + lines_dP_dp
    
    def animate(i):
        
        psis = np.load(load_psi.format(i))
        [lines_psi[p].set_ydata(psis[p]) for p in range(len(psis))]
        time_text.set_text("t = {:.2f} of {:.2f} au".format(times[i], times[-1]))
        
        # update momentum probability distribution
        if do_dP_dp_plot: 
            dPdp = np.load(load_dPdp.format(i))
            for p in range(len(psis)):
                lines_psi[len(psis)+p].set_ydata(dPdp[p])
                
        return lines_psi + [time_text,]
    
    # We'd normally specify a reasonable "interval" here...
    ani = animation.FuncAnimation(figure, animate, range(1, len(times)), 
                                  interval=int(10/dt), blit=True)
    
    
    # saving to m4 using ffmpeg writer
    plt.rcParams['animation.ffmpeg_path'] = 'C:/Users/bendikst/OneDrive - OsloMet/Dokumenter/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe' # replace wiht your local path
    writervideo = animation.FFMpegWriter(fps=30,bitrate=50000)
    ani.save(load_folder+f"/animation_{'CAP' if use_CAP else 'reg'}.mp4", dpi=500, writer=writervideo, )
    plt.show()
    
    
def run_anim_CAP(x0          = -30,
                 sigmap      = 0.1,
                 p0          = 1.9,
                 tau         = 0,
                 L           = 200,
                 n           = 512,
                 t_steps     = 200,
                 T0          = None,
                 V0          = 2,
                 w           = 1,
                 s           = 25,
                 d           = 2,
                 gamma_      = .05,
                 R_part      = .75,
                 pot_2       = 1,
                 ):

    T = T0 if T0 is not None else 2*(L/4 + np.abs(x0))/np.abs(p0)

    gamma_0 = p0 * 2 / 1000

    x = np.linspace(-L/2, L/2, n) # physical grid
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    CAP_vector, CAP_locs = square_gamma_CAP(x, dt=dt, gamma_0=gamma_0, R = R_part*L/2) # [:,0] # * 1j
    potential =  rectangular_potential(x-d, V0, s, w) + pot_2*rectangular_potential(x+d, V0, s, w)

    psis         = np.array([psi_single_initial(x,x0,sigmap,p0,tau)])
    Hamiltonians = [make_fft_Hamiltonian(n, L, dt, V = potential - sp.diags(1j*CAP_vector))[0] ]
    labels       = np.array([r"FFT $\psi$"])

    psis, Transmission, Reflection, Remainder, phi2, inte, x, k_fft = solve_once(x, psis, Hamiltonians, times, midpoint=d+w/2, 
                                                                                 CAP = [CAP_vector, CAP_locs], 
                                                                                 plot_labels=labels, V_plot=potential.diagonal(),
                                                                                 do_dP_dp_plot=True, do_live_plot=False, 
                                                                                 do_save_psi=True, do_save_dPdp=True,) 
    
    load_and_plot(use_CAP=True)
    


def run_anim_reg(x0          = -30,
                 sigmap      = 0.1,
                 p0          = 1.9,
                 tau         = 0,
                 L           = 500,
                 n           = 1024,
                 t_steps     = 200,
                 T0          = None,
                 V0          = 2,
                 w           = 1,
                 s           = 25,
                 d           = 2,
                 pot_2       = 1,
                 ):

    T = T0 if T0 is not None else (L/4 + np.abs(x0))/np.abs(p0)

    x = np.linspace(-L/2, L/2, n) # physical grid
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    potential =  rectangular_potential(x-d, V0, s, w) + pot_2*rectangular_potential(x+d, V0, s, w)

    psis         = np.array([psi_single_initial(x,x0,sigmap,p0,tau)])
    Hamiltonians = np.array([make_fft_Hamiltonian(n, L, dt, V = potential)[0]]) 
    labels       = np.array([r"FFT $\psi$"])

    
    psis, Transmission, Reflection, Remainder, phi2, inte, x, k_fft = solve_once(x, psis, Hamiltonians, times, midpoint=d+w/2, 
                                                                                 plot_labels=labels, V_plot=potential.diagonal(),
                                                                                 do_dP_dp_plot=True, do_live_plot=False, 
                                                                                 do_save_psi=True, do_save_dPdp=True,) 
    
    load_and_plot(use_CAP=False, do_dP_dp_plot=True)
    
    plt.show()



def main():    
    run_anim_CAP()
    run_anim_reg()


if __name__ == "__main__":
    main()
