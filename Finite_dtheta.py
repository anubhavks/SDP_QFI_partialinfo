# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:59:21 2023

@author: user-pc
"""

import numpy as np 
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt

def QFI(rho, G):
    Eigen_vals, eigen_vecs = np.linalg.eigh(rho)
    qfi_val = 0.0
    for i in range(len(Eigen_vals)):
        for j in range(len(Eigen_vals)):
             if Eigen_vals[i] + Eigen_vals[j] > 1e-5:
                J_z_i_j = (eigen_vecs[:, i].conjugate().transpose() @ G) @ eigen_vecs[:, j]
                qfi_val += (((Eigen_vals[i] - Eigen_vals[j])**2)/(Eigen_vals[i] + Eigen_vals[j]))*(np.abs(J_z_i_j))**2
    return 2 * qfi_val.real


#%%

N = 4  # Number of parties

#

x = np.array([[0,1],[1,0]])/2
y = np.array([[0,-1j],[1j,0]])/2
z = np.array([[1,0],[0,-1]])/2
t = np.array([[1,0],[0,1]])

def J(o,N):
    "Computes the matrix representation of the collective operator J=\sum_{i=1}^NO^{(i)} in the computational basis"
    def O(i):
        return np.kron(np.identity(2**i),np.kron(o,np.identity(2**(N-i-1))))
    return np.sum([O(i) for i in range(N)],0)

Jx = J(x,N)
Jy = J(y,N)
Jz = J(z,N)
Id = J(t,N)

#%%
G = Jz
O = [Jx + 0.2*Jz , Jy @ Jy] # The + 0.2 is to break the symmetry Jz <-> -Jz

rho_beta = sp.linalg.expm(np.einsum("i,ijk -> jk", np.random.uniform(-1,1, len(O)), O))
rho_beta /= np.trace(rho_beta)
O_ev = [np.around(np.real(np.trace(Oa @ rho_beta)),3) for Oa in O]

O_ev = [0.6071,0.5795]  # Paper
print("O_ev = ",O_ev)

#%% Verifying that Müller and Apellaniz output the same bound

def MinQFI_SDP(dtheta):
    rho = cp.Variable((len(G),len(G)), hermitian = True)
    L =  cp.Variable((len(G),len(G)), complex = True)
    rho_minus = sp.linalg.expm(+1j*dtheta*G/2) @ rho @ sp.linalg.expm(-1j*dtheta*G/2)
    rho_plus = sp.linalg.expm(-1j*dtheta*G/2) @ rho @ sp.linalg.expm(+1j*dtheta*G/2)
    constraints = [cp.trace(rho) == 1]  # rho proper density matrix
    for a in range(len(O)):
        constraints += [cp.trace(O[a] @ rho) == O_ev[a]]
    constraints += [cp.kron([[1,0],[0,0]],rho_plus) + cp.kron([[0,0],[0,1]],rho_minus) + cp.kron([[0,1],[0,0]], L) + cp.kron([[0,0],[1,0]],L.H) >> 0]  
    F = cp.Problem(cp.Maximize(cp.real(cp.trace(L))),constraints)
    F.solve(solver = "MOSEK")
    return [F.value, QFI(rho.value,G)]


Eig_min = min(np.linalg.eigvalsh(G))
Eig_max = max(np.linalg.eigvalsh(G))
mu_res = 501
mu_space = np.linspace(Eig_min,Eig_max,mu_res)
def Apellaniz():
    def wit_mu(r,mu):
        return np.max(np.linalg.eigvalsh(np.einsum("i,ijk -> jk",r,O) - 4*(G - mu*Id) @ (G - mu*Id)))
    def max_wit_mu(r):
        return max([wit_mu(r,mu) for mu in mu_space])
    f = lambda r: - (np.dot(r,O_ev) - max_wit_mu(r))
    F = sp.optimize.minimize(f, np.random.randn(len(O)), method = 'nelder-mead')  
    return  [-f(F.x), F.x]


QFI_muller = MinQFI_SDP(2*0.01*N)
QFI_apellaniz = Apellaniz()

print("QFI_muller = ", QFI_muller[1])
print("QFI_apellaniz = ", QFI_apellaniz[0])
print("discrepancy = ",  abs(QFI_muller[1]-QFI_apellaniz[0])/max(QFI_muller[1],  QFI_apellaniz[0]))

# If the discrepancy is high, try to increase the mesh mu_res

#%% Non-convexity of Apellaniz et al. approach

mu_res = 1000
mu_space = np.linspace(Eig_min,Eig_max,mu_res)

r = np.around(QFI_apellaniz[1],2)
r_rand = np.around(np.random.uniform(-1,+1,len(O)),2)
r_rand = np.around([0.89152937, 0.07584047],2) # Paper
print("r_rand = ", r_rand)

def WitApellaniz(r,mu):
    return np.dot(r,O_ev) - np.max(np.linalg.eigvalsh(np.einsum("i,ijk -> jk",r,O) - 4*(G - mu*Id) @ (G - mu*Id)))
    
wit_mu_r = [WitApellaniz(r,mu) for mu in mu_space]
wit_mu_r_rand = [WitApellaniz(r_rand,mu) for mu in mu_space]
#%%

plt.plot(mu_space/N, wit_mu_r, label = r"$\mathbf{r} = \mathbf{r}^* (optimal)$" )
plt.plot(mu_space/N, [QFI_apellaniz[0]]*mu_res,linestyle='dashed', label = r"$\min_{\mu}\mathcal{W}_\mu(\mathbf{r}^*)$")
plt.plot(mu_space/N, wit_mu_r_rand, linestyle = "dotted",label = r"$\mathbf{r}$ not optimal")
plt.xlim([-0.65/N,+0.65/N])
plt.ylim([-0.5,2])
plt.xlabel(r"$\mu/[\lambda_{\max}(\hat{G})-\lambda_{\min}(\hat{G})]$")
plt.ylabel(r"$\mathcal{W}_\mu(\mathbf{r})$")
plt.legend()
plt.savefig("Apellaniz.png", dpi = 500)



# %%   Finite dtheta

res_dtheta = 100
dtheta_max = 3
dtheta_space = np.linspace(1e-10,2*dtheta_max,res_dtheta)
muller_dtheta = np.array([MinQFI_SDP(dtheta) for dtheta in dtheta_space])


fide_dtheta = muller_dtheta[:,0]
qfi_dtheta = muller_dtheta[:,1]


#%%
# Fidelity
QFI_muller = MinQFI_SDP(2*0.02)
plt.plot(2*dtheta_space/(np.pi), fide_dtheta**2, label = r"$\mathcal{F}_{\rm max}$")
plt.plot(2*dtheta_space/(np.pi), [1-QFI_muller[1]*(i/2)**2 for i in dtheta_space], label = r"$1-F_Q[\hat{\rho}^*_{\delta\theta = 0.02},\hat{J}_z](\delta\theta)^2$", linestyle = 'dashed')
plt.plot(2*dtheta_space/(np.pi), [1-qfi_dtheta[i]*(dtheta_space[i]/2)**2 for i in range(res_dtheta)], label = r"$1-F_Q[\hat{\rho}^*_{\delta\theta},\hat{J}_z](\delta\theta)^2$", linestyle='dotted' )
plt.ylabel("Fidelity")
plt.xlabel(r"$\delta\theta/\pi $")
plt.ylim([0.85,1.01])
plt.xlim([0,4*1.4/np.pi])
plt.legend()
plt.savefig("finite_dtheta_fidelity.png", dpi = 300)

#%%
# QFI

# We disregard the first values as they are below machine's precision
fide_dtheta[0:1] = None
qfi_dtheta[0:1] = None


qfi_inverted = [4*(1-fide_dtheta[i]**2)/(dtheta_space[i]**2) for i in range(res_dtheta)]
plt.plot(2*dtheta_space/(np.pi), qfi_inverted, label = r"$(1-\mathcal{F}_{\rm max})/\delta\theta^2$" )
plt.plot(2*dtheta_space/(np.pi), [QFI_muller[1]]*res_dtheta, label = r"$F_Q[\hat{\rho}^*_{\delta\theta = 0.02},\hat{J}_z]$ ")
plt.plot(2*dtheta_space/(np.pi), qfi_dtheta, label = r"$F_Q[\hat{\rho}^*_{\delta\theta},\hat{J}_z]$",linestyle='dotted')
plt.ylabel("QFI")
plt.xlabel(r"$\delta\theta/\pi$")
plt.ylim([0.1,0.5])
#plt.xlim([0.0, 10*0.02/np.pi])
plt.xlim([0,2*1.5/(np.pi)])
plt.legend(loc = 2)
plt.savefig("finite_dtheta_QFI.png", dpi = 300)