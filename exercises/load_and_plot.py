# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 14:51:44 2023

@author: bendikst
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="dark") # nice plots


# att14 and att15 same, different gamma0
# att13 and att16 same, different gamma0



# f"TR_results/att{num}TR_single_noCAP.npy"
# f"TR_results/att{num}TR_single_CAP.npy"
# f"TR_results/att{num}TR_double_noCAP.npy"
# f"TR_results/att{num}TR_double_CAP.npy"

# num0 = "14"
# num1 = "15"

# p0s,Transmission_no,  Reflection_no,_ = np.load(f"TR_results/att{num0}TR_double_noCAP.npy", allow_pickle=True)
# p0s,Transmission_cap, Reflection_cap,_,_ = np.load(f"TR_results/att{num0}TR_double_CAP.npy", allow_pickle=True)
# p0s,Transmission_cap0,Reflection_cap0,_,_ = np.load(f"TR_results/att{num1}TR_double_CAP.npy", allow_pickle=True)

# V0 = "4.0"

# black_line = Transmission_no+Reflection_no

# # the no CAP method has a small dip in sum around 0.9
# plt.plot(p0s, Transmission_cap, label=f"WCAP {V0}") # , label="Weak CAP")
# plt.plot(p0s, Transmission_cap0, "--", label=f"SCAP  {V0}") # , label="Strong CAP")
# plt.plot(p0s, Transmission_no, "--",  label=f"NCAP  {V0}") # ,  label="No CAP")
# # plt.plot(p0s, Transmission_no+Reflection_no, "k--", label=r"$T+R$")   
# plt.plot(p0s, Transmission_cap+Reflection_cap, "k--")   
# plt.plot(p0s, Transmission_cap0+Reflection_cap0, "k--")   
# plt.xlabel(r"$p_0$")
# plt.ylabel("Probability")
# # plt.grid()
# # plt.legend()
# # plt.title("Transmission probability with double potential. V0 = 4,")
# # plt.show()

# print(np.max(np.abs(1 - Transmission_no - Reflection_no)))
# print(np.max(np.abs(1 - Transmission_cap - Reflection_cap)))
# print(np.max(np.abs(1 - Transmission_cap0 - Reflection_cap0)))


# num0 = "13"
# num1 = "16"

# p0s,Transmission_no,  Reflection_no,_ = np.load(f"TR_results/att{num0}TR_double_noCAP.npy", allow_pickle=True)
# p0s,Transmission_cap, Reflection_cap,_,_ = np.load(f"TR_results/att{num0}TR_double_CAP.npy", allow_pickle=True)
# p0s,Transmission_cap0,Reflection_cap0,_,_ = np.load(f"TR_results/att{num1}TR_double_CAP.npy", allow_pickle=True)

# V0 = 2.5

# plt.plot(p0s, Transmission_cap, label=f"WCAP {V0}") # , label="Weak CAP")
# plt.plot(p0s, Transmission_cap0, "--", label=f"SCAP  {V0}") # , label="Strong CAP")
# plt.plot(p0s, Transmission_no, "--",  label=f"NCAP  {V0}") # ,  label="No CAP")
# plt.plot(p0s, Transmission_no+Reflection_no, "k--")   
# plt.plot(p0s, Transmission_cap+Reflection_cap, "k--")   
# plt.plot(p0s, black_line, "c--") # , label=r"$T+R$ NCAP")   
# plt.plot(p0s, Transmission_cap0+Reflection_cap0, "k--") # , label=r"$T+R$") 
# plt.xlabel(r"$p_0$")
# plt.ylabel("Probability")
# plt.grid()
# plt.legend()
# plt.title("Transmission probability with double potential.") # V0 = 2.5,
# plt.savefig("report/T_double.pdf")
# plt.show()



# num0 = "14"
# num1 = "15"

# p0s,Transmission_no,  Reflection_no,_ = np.load(f"TR_results/att{num0}TR_single_noCAP.npy", allow_pickle=True)
# p0s,Transmission_cap, Reflection_cap,_,_ = np.load(f"TR_results/att{num0}TR_single_CAP.npy", allow_pickle=True)
# p0s,Transmission_cap0,Reflection_cap0,_,_ = np.load(f"TR_results/att{num1}TR_single_CAP.npy", allow_pickle=True)

# V0 = "4.0"

# plt.plot(p0s, Transmission_cap, label=f"WCAP {V0}") # , label="Weak CAP")
# plt.plot(p0s, Transmission_cap0, "--", label=f"SCAP  {V0}") # , label="Strong CAP")
# plt.plot(p0s, Transmission_no, "--",  label=f"NCAP  {V0}") # ,  label="No CAP")
# plt.plot(p0s, Transmission_no+Reflection_no, "k--")   
# plt.plot(p0s, Transmission_cap+Reflection_cap, "k--")   
# plt.plot(p0s, Transmission_cap0+Reflection_cap0, "k--") 
# plt.xlabel(r"$p_0$")
# plt.ylabel("Probability")
# # plt.grid()
# # plt.legend()
# # plt.title("Transmission probability with single potential. V0 = 4,")
# # plt.show()


# num0 = "13"
# num1 = "16"


# p0s,Transmission_no,  Reflection_no,_ = np.load(f"TR_results/att{num0}TR_single_noCAP.npy", allow_pickle=True)
# p0s,Transmission_cap, Reflection_cap,_,_ = np.load(f"TR_results/att{num0}TR_single_CAP.npy", allow_pickle=True)
# p0s,Transmission_cap0,Reflection_cap0,_,_ = np.load(f"TR_results/att{num1}TR_single_CAP.npy", allow_pickle=True)

# V0 = 2.5

# plt.plot(p0s, Transmission_cap, label=f"WCAP {V0}") # , label="Weak CAP")
# plt.plot(p0s, Transmission_cap0, "--", label=f"SCAP  {V0}") # , label="Strong CAP")
# plt.plot(p0s, Transmission_no, "--",  label=f"NCAP  {V0}") # ,  label="No CAP")
# plt.plot(p0s, Transmission_no+Reflection_no, "k--") # , label=r"$T+R$")   
# plt.plot(p0s, Transmission_cap+Reflection_cap, "k--")   
# plt.plot(p0s, Transmission_cap0+Reflection_cap0, "k--") 
# plt.xlabel(r"$p_0$")
# plt.ylabel("Probability")
# plt.grid()
# plt.legend()
# plt.title("Transmission probability with single potential.") # V0 = 2.5,
# plt.savefig("report/T_single.pdf")
# plt.show()



# cap_sing = np.load("dPdp_results/att17dP_dt_single_CAP.npy", allow_pickle=True)
# reg_sing = np.load("dPdp_results/att17dP_dt_single_noCAP.npy", allow_pickle=True)
# cap_doub = np.load("dPdp_results/att17dP_dt_double_CAP.npy", allow_pickle=True)
# reg_doub = np.load("dPdp_results/att17dP_dt_double_noCAP.npy", allow_pickle=True)

# simlocs = np.array([ np.where((np.abs(reg_doub[-2]-c) < 1e-6))[0] for c in cap_doub[-2]])[:,0]

# X,Y   = np.meshgrid(reg_doub[0], reg_doub[-2])
# X0,Y0 = np.meshgrid(cap_doub[0], cap_doub[-2])

# plt.contourf(X0,Y0, np.abs(cap_doub[-1] - reg_doub[-1][:,simlocs]).T, levels=30, alpha=1., antialiased=True)
# plt.colorbar(label="Difference")
# # plt.contourf(X, Y,  np.abs(reg_doub[-1].T), alpha=.4, antialiased=True, cmap=plt.colormaps["winter"], locator = ticker.MaxNLocator(prune = 'lower')) # , label="T double") # , norm="log")
# # plt.colorbar(label="Regular")
# plt.xlabel(r"$p_0$")
# plt.ylabel(r"$p$")
# plt.title(r"$dP/dp$ for both CAP and regular simulation with double potential.")
# plt.savefig("report/phi2_diff_double.pdf") 
# plt.show()

# plt.contourf(X0,Y0, np.abs(cap_sing[-1] - reg_sing[-1][:,simlocs]).T, levels=30, alpha=1., antialiased=True)
# plt.colorbar(label="Difference")
# # plt.contourf(X, Y,  np.abs(reg_doub[-1].T), alpha=.4, antialiased=True, cmap=plt.colormaps["winter"], locator = ticker.MaxNLocator(prune = 'lower')) # , label="T double") # , norm="log")
# # plt.colorbar(label="Regular")
# plt.xlabel(r"$p_0$")
# plt.ylabel(r"$p$")
# plt.title(r"$dP/dp$ for both CAP and regular simulation with single potential.")
# plt.savefig("report/phi2_diff_single.pdf") 
# plt.show()

# plt.contourf(X0,Y0, np.abs(cap_doub[-1] - reg_doub[-1][:,simlocs]).T/np.abs(cap_doub[-1].T), levels=30, alpha=1., antialiased=True)
# plt.colorbar(label="Relative difference")
# # plt.contourf(X, Y,  np.abs(reg_doub[-1].T), alpha=.4, antialiased=True, cmap=plt.colormaps["winter"], locator = ticker.MaxNLocator(prune = 'lower')) # , label="T double") # , norm="log")
# # plt.colorbar(label="Regular")
# plt.xlabel(r"$p_0$")
# plt.ylabel(r"$p$")
# plt.title(r"$dP/dp$ for both CAP and regular simulation with double potential.")
# plt.savefig("report/phi2_absdiff_double.pdf") 
# plt.show()

# plt.contourf(X0,Y0, np.abs(cap_sing[-1] - reg_sing[-1][:,simlocs]).T/np.abs(cap_sing[-1].T), levels=30, alpha=1., antialiased=True)
# plt.colorbar(label="Relative difference")
# # plt.contourf(X, Y,  np.abs(reg_doub[-1].T), alpha=.4, antialiased=True, cmap=plt.colormaps["winter"], locator = ticker.MaxNLocator(prune = 'lower')) # , label="T double") # , norm="log")
# # plt.colorbar(label="Regular")
# plt.xlabel(r"$p_0$")
# plt.ylabel(r"$p$")
# plt.title(r"$dP/dp$ for both CAP and regular simulation with single potential.")
# plt.savefig("report/phi2_absdiff_single.pdf") 
# plt.show()
