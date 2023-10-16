"""

To load the data and plot the figures for the One-axis twisting case.

"""

import os
import numpy as np
import matplotlib.pyplot as plt

dir = os.path.dirname(__file__)
print('Get current working directory : ', dir)

# dir = "" Enter the pathname of the file where the repository is downloaded

# To load the data stored for the linear and non-linear spin squeezing bounds
with open(dir+"Data/linearSS_10,30", "rb") as f:
# with open("non-linearSS_10,30", "rb") as f:
    N_vals = np.load(f)
    t_values = np.load(f)
    QFI_max_N = np.load(f)
    QFI_final_N = np.load(f)
    QFI_initial_N = np.load(f)
    SS_N = np.load(f)

#%%

# Plotting all the QFI, SDP bound and the spin squeezing bound for N = 10 to N = 30
for i in range(len(N_vals)):
    
    N = N_vals[i]
    plt.figure(dpi = 600)
    plt.plot(t_values/np.pi, QFI_max_N[i], label = 'Max QFI over spin generators')
    plt.plot(t_values/np.pi, QFI_initial_N[i], linestyle = "dashed", label = 'QFI initial')
    plt.plot(t_values/np.pi, QFI_final_N[i],linestyle = "dotted", label = 'SDP bound')
    plt.plot(t_values/np.pi, SS_N[i],linestyle = "--", label = r'$\xi^{-2}_{\Pi}$')
    plt.xlabel("t/π", size = 16)
    plt.ylabel("QFI/N", size = 16)
    plt.legend(fontsize = 12)
    plt.title("Linear spin squeezing for N = " + str(N_vals[i]), size = 18)
    plt.savefig(dir+"OAT_linear.png", dpi = 500)
    # plt.title("Non-linear spin squeezing for N = " + str(N_vals[i]), size = 18)
    # plt.savefig(dir+"OAT_linear.png", dpi = 500)
    plt.show()
    
#%%    

# Plotting the gap between the SDP bound obtained and the spin-squeezing bound for all N
plt.figure(dpi = 600)
for i in range(len(N_vals)):
    
    N = N_vals[i]
    plt.plot(t_values/np.pi, QFI_final_N[i] - SS_N[i], label = 'N = '+str(N_vals[i]))
    plt.title("Linear spin squeezing", size = 18)
    # plt.title("Non-linear spin squeezing", size = 18)
plt.xlabel("t/π", size = 12)
plt.ylabel("gap/N", size = 12)
plt.axhline(0, color = "black", linewidth = 0.5)
plt.legend(fontsize = 7.5)
plt.tight_layout()
plt.savefig(dir+"OAT_quad_gaps.png", dpi = 500)
plt.show()

#plt.savefig("OAT_linpar_N14.png", dpi = 500)
