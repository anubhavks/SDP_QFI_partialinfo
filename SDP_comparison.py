#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:34:19 2023
 
Comparing the different SDP methods, starting with SDP to compute Fidelity (main method), Staszek's SDP, 
and Geza Toth's implementation of SDP in  https://doi.org/10.1103/PhysRevLett.114.160501 .


"""

import numpy as np 
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
import time
import os
dir = os.path.dirname(__file__)
print('Get current working directory : ', dir)

# dir = ""     # To manually add the path name to the current directory


def spin(d):
    j = (d-1)/2
    mz = np.arange(-j, j+1)
    spi = np.diag(np.sqrt(j * (j+1) - mz[:-1] * (mz[:-1] + 1)), k = -1)
    jx = (spi+spi.T)/2
    jy = -(spi-spi.T)/(2 * 1j)
    jz = -np.diag(mz)
    return [jx, jy, jz]


def QFI(rho, G):
    Eigen_vals, eigen_vecs = np.linalg.eigh(rho)
    qfi_val = 0.0
    for i in range(len(Eigen_vals)):
        for j in range(len(Eigen_vals)):
             if Eigen_vals[i] + Eigen_vals[j] > 1e-10:
                J_z_i_j = (eigen_vecs[:, i].conjugate().transpose() @ G) @ eigen_vecs[:, j]
                qfi_val += (((Eigen_vals[i] - Eigen_vals[j])**2)/(Eigen_vals[i] + Eigen_vals[j]))*(np.abs(J_z_i_j))**2
    return 2 * qfi_val.real


def iCom(A,B):
    return 1j*(A @ B - B @ A)


def SWAP(d):
    # swaps the state of two qudits. 
    basis = np.indices([d, d]).reshape(2,-1).T
    dic = {str(basis[a]): str(a) for a in range(d**2)}
    swapped = np.array([[el[1], el[0]] for el in basis])
    indices = [int(dic[str(swapped[a])]) for a in range(d**2)]
    return np.concatenate([[np.identity(d**2)[idx]] for idx in indices])


#%%

N_max = 34 # Maximal number of parties considered.
# Such value was attainable with an ordinary laptop computer and within a significant time. However, the code 
# may crash depending on the concrete user's machine. In such case, lower N_max.  
N_values = np.arange(4, N_max)


time_gui = []
qfi_gui = []
time_stas = []
qfi_stas = []
time_toth = []
qfi_toth = []
ss = []

for N in N_values:
    print(N) # progress
    Jx, Jy, Jz = spin(N+1)
    
    G = Jx  # generator
    jy = 0.8*N/2  # mean spin
    jzjz = 0.4*N/4  # squeezed fluctuations
    constr_rho = [[Jy, jy], [Jz @ Jz, jzjz]] # constraints to be added to SDP
    
    # spin squeezing
    ss += [jy**2/jzjz]
    
    # Our SDP based on fidelity 
    dtheta = 0.01*N
    rhos = cp.Variable((len(G),len(G)), hermitian = True)
    L =  cp.Variable((len(G),len(G)), complex = True)
    rho_minus = sp.linalg.expm(+1j*dtheta*G/(2*N)) @ rhos @ sp.linalg.expm(-1j*dtheta*G/(2*N))   # That is, linear encoding
    rho_plus = sp.linalg.expm(-1j*dtheta*G/(2*N)) @ rhos @ sp.linalg.expm(+1j*dtheta*G/(2*N))
    constraints = [rhos >> 0] + [cp.trace(rhos) == 1]  # rho proper density matrix
    for con in constr_rho:
        constraints += [cp.trace(con[0] @ rhos) == con[1]]
    constraints += [cp.bmat([[rho_plus, L],[L.H, rho_minus]]) >> 0]  
    F = cp.Problem(cp.Maximize(cp.real(cp.trace(L))), constraints)
    start1 = time.time()
    F.solve(solver = "MOSEK")
    time1 = time.time()
    time_gui += [time1- start1]
    qfi_gui += [QFI(rhos.value, G)]
    #print(jy**2/jzjz, QFI(rhos.value, G))
    
    
    # Our SDP based on variance
    d = N+1 # dimension of rho
    trho = cp.Variable((d,d), hermitian = True)
    y0 = cp.Variable()
    p = cp.Variable()
    M = cp.Variable((d,d),hermitian = True)
    M2 = cp.Variable((d,d),hermitian = True)
    # Unitary encoding
    rho_out = trho
    drho_out = 1j*(G @ trho - trho @ G)
    constraints = []
    constraints.append(p == y0+cp.real(cp.trace(M)))
    constraints.append(trho>>0)
    A = cp.bmat([[rho_out,-0.5*drho_out+1j*M2],[-0.5*drho_out-1j*M2,-M]])
    constraints.append(A>>0)
    B=cp.bmat([[cp.real(cp.trace(trho)),y0],[y0,1]])
    constraints.append(B>>0)
    for con in constr_rho:
        constraints.append(cp.trace(con[0]@trho) == con[1]*cp.trace(trho))
    obj = cp.Maximize(y0+cp.real(cp.trace(M)))
    prob = cp.Problem(obj,constraints)
    start2 = time.time()
    prob.solve(solver ="MOSEK")
    time2 = time.time()
    time_stas += [time2 - start2]
    rho = trho.value
    rho /= np.trace(rho)
    qfi_stas += [QFI(rho, G)]
    print(jy**2/jzjz, QFI(rho, G), QFI(rhos.value, G))


    # SDP introduced by Tóth 
    D = N + 1
    S = SWAP(D)
    rho_t = cp.Variable((D**2, D**2), hermitian = True)
    constraints = []
    constraints.append(cp.partial_transpose(rho_t, (D, D), 0) >> 0)
    constraints.append(S @ rho_t @ S == rho_t)
    constraints.append(rho_t >> 0)
    constraints.append(cp.trace(cp.partial_transpose(rho_t, (D, D), 0)) == 1)
    for con in constr_rho:
        constraints.append(cp.trace(con[0] @ cp.partial_trace(rho_t, (D, D), 0)) == con[1])
    obj_t = cp.Minimize(cp.real(cp.trace((cp.kron(G @ G, np.eye(D)) - cp.kron(G, G)) @ rho_t)))
    F = cp.Problem(obj_t, constraints)
    start3 = time.time()
    F.solve(solver = "MOSEK")
    time3 = time.time()
    time_toth += [time3 - start3]
    rho2 = cp.partial_trace(rho_t, (D, D), 0).value
    # rho2 /= np.trace(rho2)
    qfi_toth += [QFI(rho2, G)]
    print(jy**2/jzjz, QFI(rho2, G), QFI(rhos.value, G))
    
    print("spin squeezing = " + str(jy**2/jzjz), " | bound Tóth = " + str(QFI(rho2, G)) + " | bound from variance = " + str(QFI(rho, G)) + " | bound from fidelity = " + str(QFI(rhos.value, G)))

#%%

time_gui = np.array(time_gui)
qfi_gui = np.array(qfi_gui)
time_stas = np.array(time_stas)
qfi_stas = np.array(qfi_stas)
ss = np.array(ss)

#%% Plot Figure 6a
plt.figure(dpi = 500)
plt.plot((N_values[1:] + 1)**2, time_stas[1:],"o-",label = "SDP variance")
plt.plot((N_values[1:] + 1)**2, time_gui[1:],"^--",label = "SDP fidelity")
plt.plot((N_values[1:] + 1)**2, time_toth[1:],"+-.",label = "SDP Toth")
plt.xlabel(r"$D^2 = (N+1)^2$")
plt.ylabel("solver time (s)")
plt.legend()
plt.savefig(dir+"Plots/comparison_time.png", dpi = 500)
plt.show()

#%% Plot Figure 6b
plt.figure(dpi = 500)
plt.plot((N_values[1:] + 1)**2, (qfi_gui[1:]-qfi_stas[1:])/qfi_stas[1:], "o-" ,label = "SDP variance")
plt.xlabel(r"$D^2 = (N+1)^2$")
plt.ylabel(r"$(\mathrm{QFI}_{\mathrm{f}} - \mathrm{QFI}_{\mathrm{v}})/\mathrm{QFI}_{\mathrm{v}}$")
plt.legend()
plt.savefig(dir+"Plots/comparison_preci.png", dpi = 500)
plt.show()

#%% Data
data_compGuiSta = np.array([time_gui, qfi_gui, time_stas, qfi_stas, N_values, ss]) # Save
# Please, find the precomputed data in Plots folder
data_compGuiSta = np.load(dir+"Data/data_compGuiSta.npy") # Load
time_gui, qfi_gui,time_sta, qfi_sta, N_values, ss = data_compGuiSta



