import numpy as np
import scipy as sc
import scipy.sparse as sp
import scipy.linalg as sl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time


sns.set_theme(style="dark") # nice plots

def solve(use_5_points=False,
          # various variables
          x0      = -20,
          sigmap  = 0.2,
          p0      = 3,
          tau     = 5,
          L       = 100,
          n       = 100,
          t_steps = 1000,
          T       = 20,
          n_saved = 10,
          plot_every = 10,
          ):

    use_5_points = True

    x = np.linspace(-L/2, L/2, n) # physical grid
    dt = T/t_steps
    times = np.linspace(dt, T, t_steps)

    # print(times[1]-times[0], times[2]-times[1], dt)

    # initial wave function
    psi0 = np.sqrt( np.sqrt(2) * sigmap / (np.sqrt(np.pi)*(1-2j*sigmap**2*tau)) ) * np.exp( - (sigmap**2 * (x-x0)**2 / (1-2j*sigmap**2*tau)) + 1j*p0*x)

    h = (np.max(x)-np.min(x))/n # physical step length

    # derivative matrices
    ones = np.ones(n)
    # if use_5_points:
    D2_5 = sp.diags( [-ones[2:], 16*ones[1:], -30*ones, 16*ones[1:], -ones[2:]], [-2,-1,0,1,2], format='coo') / (12*h*h)
    # else:
    D2_3 = sp.diags( [ ones[1:], -2*ones, ones[1:]] , [-1, 0, 1], format='coo') / (h*h) # second order derivative

    k_fft = 2*(np.pi/L)*np.array([i for i in range(int(n/2))]+[ j for j in range(int(-n/2),0)] ) #[i for i in range(n/2), j for j in range(-n/2,-1)]
    # print(np.array([i for i in range(int(n/2))]+[ j for j in range(int(-n/2),0)] ))
    # print(len(np.array([i for i in range(int(n/2))]+[ j for j in range(int(-n/2),0)] )))
    # exit()
    # tmp = sc.fft.fft(sp.diags(np.ones(n)).todense())
    # tmp = sc.fft.fft(np.diag(np.ones(n)))
    # exit()
    Hfft  = -1/2 * sc.fft.ifft2(sp.diags((1j*k_fft)**2) @ sc.fft.fft2(np.diag(np.ones(n))))
    # print(Hfft)
    # exit()
    # [(0:N/2-1), (-N/2:-1)]'

    H_3 = - 1/2 * D2_3 # Hamiltonian 
    H_5 = - 1/2 * D2_5 # Hamiltonian 
    # iH_3dt2 = .5j*dt*H_3 # adjusted Hamiltonian to fit the desired formula
    # iH_5dt2 = .5j*dt*H_5 # adjusted Hamiltonian to fit the desired formula
    
    exp_iH_3dt = sl.expm(-1j*H_3.todense()*dt) # adjusted Hamiltonian to fit the desired formula
    exp_iH_5dt = sl.expm(-1j*H_5.todense()*dt) # adjusted Hamiltonian to fit the desired formula
    exp_iH_fft = sl.expm(-1j*Hfft*dt)          # adjusted Hamiltonian to fit the desired formula
    
    # print(exp_iH_fft)
    # exit()

    # psis  = [psi0]
    saves = np.linspace(0, len(times)-1, n_saved).astype(int)
    # save_times = [0]

    psi_3   = psi0
    psi_5   = psi0
    psi_fft = psi0 
    # print(psi0)
    # exit()

    plt.ion()

    # here we are creating sub plots
    figure, ax = plt.subplots(figsize=(10, 8))
    line1, = ax.plot(x, np.abs(psi_3)**2, label="3-points")
    line2, = ax.plot(x, np.abs(psi_5)**2, label="5-points")
    line3, = ax.plot(x, np.abs(psi_fft)**2, label="FFT")
    line4, = ax.plot(x, psi_analytical(0, x), '--', label="Analytical")
    ax.set_ylim(top = np.max(psi_analytical(0, x))*1.1)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")
    plt.grid()
    plt.legend()
    plt.title("t = 0.")

    for t in tqdm(range(len(times))):

        # print(exp_iH_3dt.shape, psi_3.shape, psi0.shape)
        psi_3 = exp_iH_3dt @ psi_3
        psi_5 = exp_iH_5dt @ psi_5
        # print(exp_iH_fft.shape, psi_fft.shape, psi0.shape)
        # tmp = psi_fft
        psi_fft = exp_iH_fft @ psi_fft
        # print(np.max(np.abs(tmp/psi_fft)))
        # exit()
        # psi_3 = Crank_Nicolson(psi_3, iH_3dt2, dt)
        # psi_5 = Crank_Nicolson(psi_5, iH_5dt2, dt)
        if t % plot_every == 0:
            # psis.append(psi_3)
            # save_times.append(times[t])
            psi_anal = psi_analytical(times[t], x)
            # line1.set_xdata(x)
            
            line1.set_ydata(np.abs(psi_3)**2)
            line2.set_ydata(np.abs(psi_5)**2)
            line3.set_ydata(np.abs(psi_fft)**2)
            line4.set_ydata(psi_anal)
            
            plt.title(f"t = {times[t]}.")

            # drawing updated values
            figure.canvas.draw()
        
            # This will run the GUI event
            # loop until all UI events
            # currently waiting have been processed
            figure.canvas.flush_events()

            # time.sleep(0.1)
            
        # psi_3 = psi_3
        # psi_5 = psi_5
        # psi_fft = psi_fftæ

    plt.ioff()
    plt.show()
    return psi_3, psi_5, psi_anal, x, times[-1], psi0


def Crank_Nicolson(psi, F, dt):
    # a numerical approximation
    # print(F.shape, psi.shape)
    # print(np.identity(F.shape[0]))
    psi_new = ((np.identity(F.shape[0]) - F) @ psi)
    psi_new = psi_new.T
    # print(psi_new.shape)
    psi_new = np.linalg.inv(np.identity(F.shape[0]) + F) @ psi_new
    return np.ravel(psi_new)

def psi_analytical(t, x, x0 = -20, sigmap = 0.2, p0 = 3, tau = 5,): 
    N2 = np.sqrt(2/np.pi) * sigmap / np.sqrt(1+4*sigmap**4*(t-tau)**2)
    return  N2 * np.exp( - 2 * sigmap**2 * (x - x0 - p0*t)**2 / (1 + 4*sigmap**4*(t-tau)**2) )

# print(psi_new)

# psi1 = Crank_Nicolson(psi0, iHdt2, dt)
# print(psi1)

# psi2 = Crank_Nicolson(psi1, iHdt2, dt)
# print(psi2)

x0          = -20
sigmap      = 0.2
p0          = 3
tau         = 5
n           = 1024
t_steps     = 1000
T           = 35
# n_saved     = 20
plot_every  = 10

psis_3, psis_5, psi_anal, x, time_final, psi0 = solve(use_5_points=False, x0=x0, sigmap=sigmap, p0=p0, tau=tau, T=T, n=n, t_steps=t_steps, plot_every=plot_every)
# psis_5, x, save_times = solve(use_5_points=True , x0=x0, sigmap=sigmap, p0=p0, tau=tau, T=T, n=n, t_steps=t_steps, n_saved=n_saved)

# plot_number = -1
# t = save_times[plot_number]

# psi_anal = psi_analytical(t, x)
# plt.clf()

# figure0, ax0 = plt.subplots(figsize=(10, 8))

# # ax0.plot(x, np.abs(psi0)**2 , '--', label=f"t = {0}")
# ax0.plot(x, np.abs(psis_3)**2, label="3-points")
# ax0.plot(x, np.abs(psis_5)**2, label="5-points")
# ax0.plot(x, psi_anal, '--', label="Analytical")
# plt.legend()
# plt.title(f"t = {time_final}.")
# plt.grid()
# plt.xlabel(r"$x$")
# plt.ylabel(r"$\left|\Psi\left(x \right)\right|^2$")

# figure0.canvas.draw()
# plt.show()

