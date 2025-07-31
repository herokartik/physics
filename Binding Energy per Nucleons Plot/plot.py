import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("data.csv")


A=df["mass_no"]
Z=df["atomic_no"]
N=A-Z


#values of parameters
a_v=14.1
a_s=13.0
a_c=0.595
a_asy=19.0
a_p=33.5


#Volume energy term
T1=a_v*A
#Surface energy term
T2=a_s*A**(2/3)
#Coulumb energy term
T3=a_c*Z*(Z-1)*A**(-1/3)
#Asymetric energy term
T4=a_asy*((A-2*Z)**2)/A


#For Pairing energy
#---------------
P=[]

for z,n,a in zip(Z,N,A):
    if z%2==0 and n%2==0:
        P.append(+a_p*a**(-4/3))  #Even-Even Interaction
    elif z%2==1 and n%2==1:
        P.append(-a_p*a**(-4/3))  #Odd-Odd Interaction
    else:
        P.append(0) #Other Interaction
P=pd.Series(P)


#--------------
#Total binding energy
BE1=T1-T2-T3-T4+P


#Binding energy per nucleon
BEA=BE1/A

max_bea = BEA.max()
max_A = A[BEA.idxmax()]
print(f'{max_bea} & {max_A}')

plt.figure(1)
plt.plot(A,BEA)
plt.xlabel(r'Mass number $A$')
plt.ylabel("Binding energy per nucleon (MeV)")
plt.title(r'$\frac{BE}{A}$ vs $A$')
plt.grid(True)
plt.savefig("binding_energy_per_nucleon.png", dpi=300)

plt.figure(2)
plt.plot(Z,N)
plt.xlabel("Z (Atomic Number)")
plt.ylabel("N ( Number of Neutrons)")
plt.title("Z Vs N")
plt.grid(True)
plt.savefig("Z vs N.png",dpi=300)
plt.show()


