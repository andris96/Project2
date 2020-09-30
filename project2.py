from  matplotlib import pyplot as plt
import numpy as np

#opening file
infile = open("Eigenvector.txt","r")

#saving each line in an array called lines
lines = infile.readlines()
n = len(lines)

v = np.zeros(n)
x = np.linspace(0,1,n)

#storing each element in array
for i in range(n):
    v[i] = float(lines[i])

v_analytical = np.zeros(n)
#calculating the analytical values of the eigenvector
for i in range(n):
    v_analytical[i] = np.sin(np.pi*i/(n+1))

v_analytical = v_analytical/np.linalg.norm(v_analytical)

#plotting
plt.plot(x,v, label = 'numerical eigenvector')
plt.plot(x,v_analytical, label = 'analytical eigenvector')
plt.xlabel('x')
plt.ylabel('v')
plt.legend()
plt.show()