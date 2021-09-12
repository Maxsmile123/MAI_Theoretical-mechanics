import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

# Var 25: r(t) = cos(3t), phi(t) = t

def Rot2D(X, Y, Alpha):
    RX = X*np.cos(Alpha) - Y*np.sin(Alpha)
    RY = X*np.sin(Alpha) + Y*np.cos(Alpha)
    return RX, RY

t = sp.Symbol('t')
R = 4
Omega = 2


x = sp.cos(3*t)*sp.cos(t)
y = sp.cos(3*t)*sp.sin(t)

Vx = sp.diff(x, t)
print(Vx)

Vy = sp.diff(y, t)
print(Vy)


Vmod = sp.sqrt(Vx*Vx+Vy*Vy); # Скорость общая
Wx = sp.diff(Vx, t) # Ускорение по x
print(Wx)
Wy = sp.diff(Vy, t)
print(Wy)
Wmod = sp.sqrt(Wx*Wx+Wy*Wy);
#and here really we could escape integrating, just don't forget that it's absolute value of V here we should differentiate
Wtau = sp.diff(Vmod,t)
#this is the value of rho but in the picture you should draw the radius, don' t forget!
rho = (Vmod*Vmod)/sp.sqrt(Wmod*Wmod-Wtau*Wtau)
xC = R*Omega*t

#constructing corresponding arrays
T = np.linspace(0, 10, 1000)
# Зануление всех элементов массива
X = np.zeros_like(T)
Y = np.zeros_like(T)
VX = np.zeros_like(T)
VY = np.zeros_like(T)
XC = np.zeros_like(T)
YC = R
#filling arrays with corresponding values
for i in np.arange(len(T)):
    X[i] = sp.Subs(x, t, T[i])
    Y[i] = sp.Subs(y, t, T[i])
    VX[i] = sp.Subs(Vx, t, T[i])
    VY[i] = sp.Subs(Vy, t, T[i])
    XC[i] = sp.Subs(xC, t, T[i])

#here we start to plot
fig = plt.figure()

ax1 = fig.add_subplot(1, 1, 1)
ax1.axis('equal')
ax1.set_xlim([-2, 2])
ax1.set_ylim([-1, 1])

#plotting a trajectory
ax1.plot(X, Y)

#plotting a plane for a disc
ax1.plot([X.min(), X.max()], [0, 0], 'black')

#plotting initial positions

#of the point A on the disc
P, = ax1.plot(X[0], Y[0], marker='o')
#of the velocity vector of this point (line)
VLine, = ax1.plot([X[0], X[0]+VX[0]], [Y[0], Y[0]+VY[0]], 'r')

#of the velocity vector of this point (arrow)
ArrowX = np.array([-0.2*R, 0, -0.2*R])
ArrowY = np.array([0.1*R, 0, -0.1*R])
RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[0], VX[0]))
VArrow, = ax1.plot(RArrowX + X[0]+VX[0], RArrowY + Y[0]+VY[0])


#function for recounting the positions
def anima(i):
    P.set_data(X[i], Y[i])
    VLine.set_data([X[i], X[i]+VX[i]], [Y[i], Y[i]+VY[i]])
    RArrowX, RArrowY = Rot2D(ArrowX, ArrowY, math.atan2(VY[i], VX[i]))
    VArrow.set_data(RArrowX + X[i]+VX[i], RArrowY + Y[i]+VY[i])

    return P, VLine, VArrow
# animation function
anim = FuncAnimation(fig, anima, frames=1000, interval=2, blit=True)

plt.show()

