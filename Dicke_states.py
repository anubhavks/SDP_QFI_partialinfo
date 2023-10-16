# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 16:05:12 2023

@author: user-pc
"""

"""

Applying the SDP to the case of Dicke state metrology

"""

import numpy as np 
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt

import os
dir = os.path.dirname(__file__)
print('Get current working directory : ', dir)

# dir = ""     # To manually add the path name to the current directory

#%%


def spin(d):
    "Computes d x d spin-j matrices (d = 2j + 1)"
    j = (d-1)/2
    mz = np.arange(-j, j+1)
    spi = np.diag(np.sqrt(j * (j+1) - mz[:-1] * (mz[:-1] + 1)), k = -1)
    jx = (spi+spi.T)/2
    jy = -(spi-spi.T)/(2 * 1j)
    jz = -np.diag(mz)
    return [jx, jy, jz]

def QFI(rho, G):
    "Definition of quantum Fisher information with respect generator G (unitary encoding)"
    Eigen_vals, eigen_vecs = np.linalg.eigh(rho)
    qfi_val = 0.0
    for i in range(len(Eigen_vals)):
        for j in range(len(Eigen_vals)):
            if Eigen_vals[i] + Eigen_vals[j] > 1e-10:
                J_z_i_j = (eigen_vecs[:, i].conjugate().transpose() @ G) @ eigen_vecs[:, j]
                qfi_val += (((Eigen_vals[i] - Eigen_vals[j])**2)/(Eigen_vals[i] + Eigen_vals[j]))*(np.abs(J_z_i_j))**2
    return 2 * qfi_val.real

#%%

N = 16  # Number of parties

#

D_list = [N+1-2*int(k) for k in range(1 + N//2)] # Dimensions of the permutation-invariant blocks
J_list = [spin(d) for d in D_list]
J_vect = spin(2)

def J_PI(J):
    "Computes the matrix representation of the collective operator J=\sum_{i=1}^No^{(i)} in the coarse-grained permutation-invariant basis"
    coefs = [np.trace(J @ Ja)/np.trace(Ja @ Ja) for Ja in J_vect]
    res_blocks = [np.sum([coefs[v] * J_list[s][v] for v in range(len(J_list[s]))],0) for s in range(len(J_list))]
    return sp.linalg.block_diag(*(res_blocks))
 
Jx = J_PI(J_vect[0])
Jy = J_PI(J_vect[1])
Jz = J_PI(J_vect[2])

def full_to_block(A):
    return [[np.array([[a[l + sum(D_list[0:j]), k + sum(D_list[0:j])] for k in range(D_list[j])] for l in range(D_list[j])], dtype = complex) for j in range(len(D_list))] for a in A]

Jx_block = full_to_block([Jx])[0]
Jy_block = full_to_block([Jy])[0]
Jz_block = full_to_block([Jz])[0]

G = Jz
G_block = full_to_block([G])[0]

#%% DICKE SQUEEZING  

res = 30  # resolution of the figure

# Data
Jsq = 0.7*(N/2)*((N/2)+1)
JxJx_values = np.linspace(0,2,res)
JyJyPJzJz_values = Jsq - np.linspace(0,2,res)


QFI_final = []  # Our SDP bound
SS = [] # Dicke spin-squeezing bound
for i in range(res):
    print(i/res)  # Progress
    JxJx_ev = JxJx_values[i]
    JyJyPJzJz_ev = JyJyPJzJz_values[i]
    SS_i = np.real(JyJyPJzJz_ev/(N*(JxJx_ev + 1/4)))/2
 
    # SDP algorithm
    # Variables
    rho_block = [cp.Variable((d, d), hermitian = True) for d in D_list] # State
    L_block = [cp.Variable((d, d), complex = True) for d in D_list]
    # Constraints
    constraints = [cp.sum([cp.trace(rho_block[j]) for j in range(len(D_list))]) == 1] # Normalization
    constraints += [cp.sum([cp.trace(Jx_block[j] @ Jx_block[j] @ rho_block[j]) for j in range(len(D_list))]) == JxJx_ev] 
    constraints += [cp.sum([cp.trace((Jy_block[j] @ Jy_block[j] + Jz_block[j] @ Jz_block[j]) @ rho_block[j]) for j in range(len(D_list))]) == JyJyPJzJz_ev]
    constraints += [cp.sum([cp.trace(Jx_block[j] @ rho_block[j]) for j in range(len(D_list))]) == 0] # Zero mean spin 
    constraints += [cp.sum([cp.trace(Jy_block[j] @ rho_block[j]) for j in range(len(D_list))]) == 0] 
    constraints += [cp.sum([cp.trace(Jz_block[j] @ rho_block[j]) for j in range(len(D_list))]) == 0]  
    
    dtheta = 0.1
    rho_block_minus = [sp.linalg.expm(+1j * dtheta * G_block[j] / (2 * N)) @ rho_block[j] @ sp.linalg.expm(-1j * dtheta * G_block[j] / (2 * N)) for j in range(len(D_list))]   
    rho_block_plus = [sp.linalg.expm(-1j * dtheta * G_block[j] / (2 * N)) @ rho_block[j] @ sp.linalg.expm(+1j * dtheta * G_block[j] / (2 * N)) for j in range(len(D_list))]
    for j in range(len(D_list)):
        constraints += [cp.kron([[1,0],[0,0]],rho_block_plus[j]) + cp.kron([[0,0],[0,1]],rho_block_minus[j]) + cp.kron([[0,1],[0,0]], L_block[j]) + cp.kron([[0,0],[1,0]], L_block[j].H) >> 0]
       
    F = cp.Problem(cp.Maximize(cp.sum([cp.real(cp.trace(L_block[j])) for j in range(len(D_list))])), constraints)
    F.solve(solver = "MOSEK")
        
    rho_final_block = [rho_block[j].value for j in range(len(D_list))]
    rho_final = sp.linalg.block_diag(*([rho_block[j].value for j in range(len(D_list))]))

    if rho_final[0][0] is not None:   
        QFI_i = QFI(rho_final, G)/N
    else:  # If the data would not be compatible with any state
        QFI_i = None
        SS_i = None
    QFI_final += [QFI_i]
    SS += [SS_i]
    
#%% Load instead the precomputed data
data_Dicke_param = np.load(dir+"/Data/data_Dicke_param.npy")
x = data_Dicke_param[0]
SS = data_Dicke_param[1]
QFI_final = data_Dicke_param[2]

#%% Plot Figure 1 

# #plt.rcParams.update({"font.size": 12})

# x = JxJx_values[0:len(QFI_final)]
# plt.plot(x, SS, linestyle = "dashed", label = r"$(2\xi_D)^{-1}$")
# plt.plot(x, QFI_final, label = "SDP bound")
# plt.xlabel(r"$<\hat{J}_x^2>$")
# plt.ylabel(r'$\mathrm{QFI}_{\hat{J}_z}/N$')
# plt.legend()
# #plt.tight_layout()
# plt.savefig(dir+"/Plots/Dicke_param.png", dpi = 500)


# #data_Dicke_param = np.array([x, SS, QFI_final])
# #np.save(dir+"/Data/data_Dicke_param.npy", data_Dicke_param)


#%%  SCALING

N_max = 70  # Maximal number of particles that we reach

a_list = [0.01,0.02,0.03,0.05,0.08,0.1]
N_list = [2*(i+1) for i in range(int(N_max/2))]
QFI_final = []
for a in a_list:
    QFI_final_a = []
    for N in N_list:
        print("progress, N =", N," out of", max(N_list))
        
        Jx = spin(N+1)[0]
        Jy = spin(N+1)[1]
        Jz = spin(N+1)[2]
        
        G = Jz  # Generator
    
        # Since the collective spin is maximal, we can restrict to the symmetric sector
        rho = cp.Variable((N+1, N+1), hermitian = True) 
        L = cp.Variable((N+1, N+1), complex = True)
        
        constraints = [cp.trace(rho) == 1] 
        constraints += [cp.trace(Jx @ Jx @ rho) == a*(N/4)]
        constraints += [cp.trace(Jx  @ rho) == 0]
        
        dtheta = 0.1
        rho_minus = rho + 1j*dtheta*(rho @ G - G @ rho)/(2*N)
        rho_plus = rho - 1j*dtheta*(rho @ G - G @ rho)/(2*N)
        constraints += [cp.kron([[1,0],[0,0]],rho_plus) + cp.kron([[0,0],[0,1]],rho_minus) + cp.kron([[0,1],[0,0]], L) + cp.kron([[0,0],[1,0]],L.H) >> 0]  
        F = cp.Problem(cp.Maximize(cp.real(cp.trace(L))),constraints)
        F.solve(solver = "MOSEK")
        QFI_final_a += [QFI(rho.value,G)/N]          
    QFI_final += [QFI_final_a]  

#%% Saves values 
with open(dir+"/Data/dicke_states_N_70", "wb") as f:
    np.save(f, N_list)
    np.save(f, a_list)
    np.save(f, QFI_Dicke)
    np.save(f, QFI_final)

#%% Computation takes a while, we provide precomputed values
with open(dir+"/Data/dicke_states_N_70", "rb") as f_load:
    N_list = np.load(f_load)
    a_list = np.load(f_load)
    QFI_Dicke = np.load(f_load)
    QFI_final = np.load(f_load)
            
    
#%% Figure 8a

QFI_Dicke = [(i/2+1) for i in N_list]

plt.figure(dpi = 500)
plt.scatter(N_list, QFI_Dicke, marker = "o", label = "a = 0 ")
plt.scatter(N_list, QFI_final[0],marker = "+", label = "a = 0.01")
plt.scatter(N_list, QFI_final[1],marker = "*", label = "a = 0.02")
plt.scatter(N_list, QFI_final[2],marker = "v", label = "a = 0.03")
plt.scatter(N_list, QFI_final[3],marker = "d", label = "a = 0.05")
plt.scatter(N_list, QFI_final[4],marker = "p", label = "a = 0.10")
plt.xlabel(r"$N$")
plt.ylabel(r'$\mathrm{QFI}_{\hat{J}_z}/N$')
plt.legend()
plt.tight_layout()
plt.savefig(dir+"/Plots/Dicke_N.png", dpi = 500)
plt.show()

#%% Figure 8b

plt.figure(dpi = 500)
plt.plot(N_list/4*a_list[0], QFI_final[0],marker = "+", label = "a = 0.01", color = "C1")
plt.plot(N_list/4*a_list[1], QFI_final[1],marker = "*", label = "a = 0.02", color = "C2")
plt.plot(N_list/4*a_list[2], QFI_final[2],marker = "v", label = "a = 0.03", color = "C3")
plt.plot(N_list/4*a_list[3], QFI_final[3],marker = "d", label = "a = 0.05", color = "C4")
plt.plot(N_list/4*a_list[4], QFI_final[4],marker = "p", label = "a = 0.10", color = "C5")
plt.vlines(0.25, 0, np.max(QFI_final), color = "black", linestyles = "--")
plt.xlabel(r"$\langle\hat{J}_x^2\rangle$")
plt.ylabel(r'$\mathrm{QFI}_{\hat{J}_z}/N$')
plt.legend()
plt.tight_layout()
plt.savefig(dir+"/Plots/Dicke_Jx.png", dpi = 500)
plt.show()

