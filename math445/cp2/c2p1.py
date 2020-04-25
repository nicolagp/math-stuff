import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# Global parameters

N = 51 #number of grid points
dt = 5.e-4 #time step
L = float(1) #size of grid
nsteps = 620 #number of time steps
dx = L/(N-1) #grid spacing
nplot = 20 #number of timesteps before plotting

r = (0.25 * dt )/ dx**2

# initialize matrices A, B and b array
A = np.zeros((N-2,N-2))
B = np.zeros((N-2,N-2))

#define matrices A, B
for i in range(N-2):
    if i==0:
        A[i,:] = [2+2*r if j==0 else (-r) if j==1 else 0 for j in range(N-2)]
        B[i,:] = [2-2*r if j==0 else r if j==1 else 0 for j in range(N-2)]
    elif i==N-3:
        A[i,:] = [-r if j==N-4 else 2+2*r if j==N-3 else 0 for j in range(N-2)]
        B[i,:] = [r if j==N-4 else 2-2*r if j==N-3 else 0 for j in range(N-2)]
    else:
        A[i,:] = [-r if j==i-1 or j==i+1 else 2+2*r if j==i else 0 for j in range(N-2)]
        B[i,:] = [r if j==i-1 or j==i+1 else 2-2*r if j==i else 0 for j in range(N-2)]

#initialize grid
x = np.linspace(0,1,N)
t = np.linspace(0,2,N)
#initial conditions (t=0)
u = np.asarray([20*xx if xx<=0.5 else 20*(1-xx) for xx in x])
#evaluate right hand side at t=0
rhs = B.dot(u[1:-1])

# solution matrix
sol = np.zeros((N, N))

# calculate solution for each time step
for j in range(N):
	sol[1:-1, j] = np.linalg.solve(A,rhs)

	# update rhs
	rhs = B.dot(sol[1:-1, j])

# plotting
fig = plt.figure()
ax = plt.axes(projection='3d')
x, t = np.meshgrid(x, t)

ax.plot_surface(x, t, sol, rstride=1, cstride=1,
                cmap='coolwarm')
ax.set_title('$u(x, t)$')
ax.set_xlabel("$x$")
ax.set_ylabel("$t$")
ax.set_zlabel("$u$")
plt.tight_layout()

fig.savefig("cn_heat.png")