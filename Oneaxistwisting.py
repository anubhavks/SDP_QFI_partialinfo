# -*- coding: utf-8 -*-
"""

One-axis twisting metrology

"""

import numpy as np 
import cvxpy as cp
import scipy as sp
import matplotlib.pyplot as plt
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


#%%

N = 20  # Number of parties

#%%


# Spin
Jx = spin(N+1)[0]   
Jy = spin(N+1)[1]   
Jz = spin(N+1)[2]  

# Parity

Piz =  np.diag([(-1)**a for a in range(N+1)])
Pix =  sp.linalg.expm(1j*np.pi*Jy/2) @ Piz @ sp.linalg.expm(-1j*np.pi*Jy/2)
Piy =  sp.linalg.expm(1j*np.pi*Jx/2) @  Piz @ sp.linalg.expm(-1j*np.pi*Jx/2)

    
#%% Choose 
 
rho_init = np.zeros((np.shape(Jz)[0], np.shape(Jz)[1]))
rho_init[0,0] = 1                                  # Initial state: CSSz

t_values = np.concatenate((np.linspace(1e-5*np.pi, 0.99*np.pi/2, 200),[np.pi/3, np.pi/4]))  # Time 
t_values = np.sort(t_values)

H = Jx @ Jx   # Hamiltonian: OAT

O_list_lin = [Jx,Jy,Jz]     # Span generator and measurements
# Span measurements
# # Parity measurements
#O_list_nonlin = [Pix, Piy, Piz]  
#O_list_nonlin = [sp.linalg.expm(-1j*Jy*α) @ Piz @ sp.linalg.expm(1j*Jy*α)]
#O_list_nonlin = [Pix, Piy, Piz]
#O_list_nonlin = [Piz, 1j*(Piz @ Jy - Jy @ Piz), Pix] 
# Spin moments
#O_list_nonlin = [] # Linear
O_list_nonlin = [Jx @ Jx, Jy @ Jy, Jz @ Jz, Jx @ Jy + Jy @ Jx, Jx @ Jz + Jz @ Jx, Jy @ Jz + Jz @ Jy] # Quadratic
 
O_list = O_list_lin + O_list_nonlin


#%%
SS = []   # Squeezing lower bound with optimized generator G <=
QFI_final = []  # <= Our SDP-based sensitivity bound with respect the same generator G <=
QFI_initial = [] # <= QFI of the OAT state for G <=
QFI_max = [] # <= Max QFI over linear spin generators G given the OAT state

for t in t_values:
    print(t/max(t_values)) # Progress
    
    rhot = sp.linalg.expm(+1j*t*H) @ rho_init @ sp.linalg.expm(-1j*t*H) # time evolution
    rhot = rhot + 1e-4*np.identity(N+1)/(N+1) # add infinitessimal mixing to be in the safe side
    rhot /= np.trace(rhot)
    

    # Generalized squeezing bound following Gessner derivation
    Gamma =  np.array([[2*np.imag(np.trace(rhot @ Oa @ Ob)) for Oa in O_list] for Ob in O_list])
    ExpVals = np.array([[np.real(np.trace(rhot @ Oa)) for Oa in O_list]])
    SecMoms = np.array([[np.real(np.trace(rhot @ Oa @ Ob)) for Oa in O_list] for Ob in O_list])
    CoVar  = np.array(SecMoms - np.transpose(ExpVals) @ ExpVals)
    M = np.transpose(Gamma) @ np.linalg.pinv(CoVar) @ Gamma 
    Mtilde = M[0:len(O_list_lin),0:len(O_list_lin)]
    evals, evects = np.linalg.eigh(Mtilde)
    SS_t = evals[-1]/N
    # Optimal generator for squeezing
    n = evects[:,-1]
    G = np.sum([n[k]*O_list_lin[k] for k in range(len(O_list_lin))], axis = 0)  
        
    # SDP  
    rho = cp.Variable((N+1, N+1), hermitian = True)
    L = cp.Variable((N+1, N+1), complex = True) 
    constraints = [cp.trace(rho) == 1] 
    for a in range(len(O_list)):
        constraints += [cp.real(cp.trace(O_list[a] @ rho))  ==  np.real(np.trace(O_list[a] @ rhot))]  
        for b in range(len(O_list)):
                constraints += [cp.real(cp.trace(O_list[a] @ O_list[b] @ rho))  ==  np.real(np.trace(O_list[a] @ O_list[b] @ rhot))]
    
    dtheta = 0.1/N
    rho_minus = sp.linalg.expm(1j*G*dtheta) @ rho @ sp.linalg.expm(-1j*G*dtheta)
    rho_plus = sp.linalg.expm(-1j*G*dtheta) @ rho @ sp.linalg.expm(1j*G*dtheta)
    constraints += [cp.kron([[1,0],[0,0]],rho_plus) + cp.kron([[0,0],[0,1]],rho_minus) + cp.kron([[0,1],[0,0]], L) + cp.kron([[0,0],[1,0]],L.H) >> 0]  
    
    F = cp.Problem(cp.Minimize(cp.real(cp.trace(L))), constraints)
    F.solve(solver = "MOSEK")

    rho_final = rho.value

    
    QFI_final += [QFI(rho_final, G)/N]
    QFI_initial += [QFI(rhot, G)/N]
    SS  += [SS_t]
    QFI_max += [4*np.max(np.linalg.eigvalsh(CoVar[0:3,0:3]))/N] 
    

#%%

# By selecting different measurement strategies Figures 2,3,4 can be reproduced. 
# We can also study the scaling by increasing N (Figure 9)
# All data is contained in the "data" folder and the plots can be generated from "Plots_data.py"
plt.figure(dpi = 500)
plt.plot(t_values/np.pi, QFI_max, label = 'Max QFI over spin generators')
plt.plot(t_values/np.pi, QFI_initial,linestyle = "dashed", label = 'QFI initial')
plt.plot(t_values/np.pi, QFI_final,linestyle = "dotted", label = 'SDP bound')
plt.plot(t_values/np.pi, np.array(SS),linestyle = "--", label = r'$\xi^{-2}_{\Pi}$')
plt.xlabel("t/π")
plt.ylabel("QFI/N")
plt.legend()
#plt.savefig(dir+"Plots/OAT_linpar_N14.png", dpi = 500) # save figure


