# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 12:02:19 2022

@author: user-pc
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp
import cvxpy as cp
import os
dir = os.path.dirname(__file__)
print('Get current working directory : ', dir)

# dir = ""     # To manually add the path name to the current directory

# Pauli matrices
x = np.array([[0,1],[1,0]])/2
y = np.array([[0,-1j],[1j,0]])/2
z = np.array([[1,0],[0,-1]])/2

def QFI(rho, G):
    "Definition quantum Fisher information with respect generator G (unitary encoding)"
    Eigen_vals, eigen_vecs = np.linalg.eigh(rho)
    qfi_val=0.0
    for i in range(len(Eigen_vals)):
        for j in range(len(Eigen_vals)):
            if Eigen_vals[i]+Eigen_vals[j]>0:
                J_z_i_j = (eigen_vecs[:, i].conjugate().transpose() @ G) @ eigen_vecs[:, j]
                qfi_val += (((Eigen_vals[i] - Eigen_vals[j])**2)/(Eigen_vals[i]+Eigen_vals[j]))*(np.abs(J_z_i_j))**2
    return 2*qfi_val.real

#%%

N = 4  # Number of particles

def Op(o,i):
    # Applies operator o at party i
    d = 2
    return np.kron(np.kron(np.identity(d**i), o), np.identity(d**(N-1-i)))

      
def Op_k(o,k):
    # Collective operator at momentum k
    d = 2**N
    res = np.zeros((d,d), dtype = complex)
    for i in range(N):
        res += Op(o,i)*np.exp(1j*2*np.pi*i*k/N)
    return res

#%%

H = np.zeros((2**N,2**N), dtype = complex)  # Hamiltonian
for i in range(N-1):
    H += Op(x,i) @ Op(x,i+1) 

psi_0 =  np.identity(2**N)[0,:]  # Initial state
    
Md = [[Op_k(x,k), Op_k(y,k), Op_k(z,k)] for k in range(N)]     # Observables
    
#%%

res = 50  # resolution time step
t_space = np.linspace(0.01, 2*np.pi, res)

QFI_max = []    # max QFI over generators G given state
QFI_psi = []    # QFI of the state with G = Jas (generator aligned with the antisqueezed component) 
QFI_SDP = []    # Our SDP bound w.r.t. G = Jas given data
for t in t_space:
    print(t/max(t_space))
    rhot = sp.expm(1j*t*H)  @ np.outer(psi_0,psi_0) @ sp.expm(-1j*t*H) # time evolution
    
    # Computation optimal generator for spin-squeezing (momentum zero), Jas = G
    O_list = Md[0]
    Gamma =  np.array([[2*np.imag(np.trace(rhot @ Oa @ Ob)) for Oa in O_list] for Ob in O_list])
    ExpVals = np.array([[np.real(np.trace(rhot @ Oa)) for Oa in O_list]])
    SecMoms = np.array([[np.real(np.trace(rhot @ Oa @ Ob)) for Oa in O_list] for Ob in O_list])
    CoVar  = np.array(SecMoms - np.transpose(ExpVals) @ ExpVals)
    M = np.transpose(Gamma) @ np.linalg.pinv(CoVar) @ Gamma 
    Mtilde = M[0:len(O_list),0:len(O_list)]
    evals, evects = np.linalg.eigh(Mtilde)
    #SS_t = evals[-1]/N 
    n = evects[:,-1]
    G = np.sum([n[k]*O_list[k] for k in range(len(O_list))], axis = 0)  
   
    
    # SDP
    dtheta = 1
    tmp = []
    for k in range(N):
        rho = cp.Variable((len(G),len(G)), hermitian = True)
        L =  cp.Variable((len(G),len(G)), complex = True)
        rho_minus = sp.expm(+1j*dtheta*G/(2*N)) @ rho @ sp.expm(-1j*dtheta*G/(2*N))
        rho_plus = sp.expm(-1j*dtheta*G/(2*N)) @ rho @ sp.expm(+1j*dtheta*G/(2*N))
        constraints = [cp.trace(rho) == 1]  # rho proper density matrix
        for M1 in Md[0]:
            constraints += [cp.trace(M1 @ rho)  == np.trace(M1 @ rhot)]  # First moments
        for r in range(k+1):  # We incorporate data up to momentum k
            for M1 in Md[r]:
                for M2 in Md[r]:
                    constraints += [cp.real(cp.trace(M1 @ np.transpose(np.conjugate(M2)) @ rho))  == np.real(np.trace(M1 @ np.transpose(np.conjugate(M2)) @ rhot))]  # Second moments
        constraints += [cp.bmat([[rho_plus, L],[L.H, rho_minus]]) >> 0]
        F = cp.Problem(cp.Maximize(cp.real(cp.trace(L))),constraints)
        F.solve(solver = "MOSEK")  
        tmp += [QFI(rho.value,G)/N]
    QFI_SDP += [tmp]
    QFI_psi += [QFI(rhot,G)/N]       
    QFI_max += [4*np.max(np.linalg.eigvalsh(CoVar))/N]
    print(tmp,QFI(rhot,G))

QFI_SEP = np.array(QFI_SEP)
#%%
plt.figure(dpi = 500)
plt.plot(t_space/np.pi, QFI_max, label = "QFI max")
plt.plot(t_space/np.pi, QFI_psi, label = "QFI psi")
plt.plot(t_space/np.pi, QFI_SDP[:,0], label = "K = 0",linestyle = "dashed")
plt.plot(t_space/np.pi, QFI_SDP[:,1], label = "K ≥ 1",linestyle = "dotted")
plt.plot(t_space/np.pi, QFI_SDP[:,3], label = "K ≥ 1",linestyle = "dotted")
plt.xlabel(r"$t/\pi$")
plt.ylabel("QFI/N$")
plt.legend()
# plt.savefig(dir+"Plots/K.png", dpi = 500)  # Uncomment to save the figure.
plt.show()
