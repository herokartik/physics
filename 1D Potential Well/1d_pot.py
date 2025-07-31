import numpy as np
import matplotlib.pyplot as plt

a=1 #length of well
h=1 #hbar
m=1 #mass
N=1000 #no of points
n=7 #energy state


x=np.linspace(-a/2,a/2,N) #symmetric potential well
dx=x[1]-x[0]

D2=-2*np.diag(np.ones(N))+np.diag(np.ones(N-1),-1)+np.diag(np.ones(N-1),1) #2nd diff operator in finite difference method

T=-(h**2/2*m*dx**2)*D2 #kinetic
V=np.zeros(N) #potential
H=T+V #hamiltonia

E,psi=np.linalg.eigh(H)

psi=psi[:,n]
#To normalise the wave function
prob_density=np.abs(psi**2)
norm=np.trapz(prob_density,x)

norm_psi=psi/np.sqrt(norm)


fig,(ax1,ax2)=plt.subplots(2,1,sharex=True)
fig.suptitle(f"Wave function and it's probability density at n={n}")
ax1.plot(x,norm_psi)
ax1.set_ylabel(r"$\psi$")
ax1.grid()
ax2.plot(x,prob_density)
ax2.set_ylabel(r"$|\psi|^2$")
ax2.grid()

plt.savefig('1d_inf_well.png',dpi=400)
plt.show()


