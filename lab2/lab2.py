import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Circle(X, Y):
    CX = [X + R * math.cos(i/100) for i in range(0, 628)]
    CY = [Y + R * math.sin(i/100) for i in range(0, 628)]
    return CX, CY

def CircleB(X, Y):
    CX = [X + r * math.cos(i/100) for i in range(0, 628)]
    CY = [Y + r * math.sin(i/100) for i in range(0, 628)]
    return CX, CY

def anima(i):
    Beam_AO, = ax1.plot([R, R -XO[i]], [0, -YO[i]], 'r')
    Beam_CB, = ax1.plot([R + OC, R + OC + XB[i]], [-R, -R - YB[i]], 'g')
    circleB, = ax1.plot(*CircleB(R + OC + XB[i], -R - YB[i]), 'g')
    return Beam_AO, Beam_CB, circleB,


BC = 4
OC = 5
R = 1.75
r = 0.35

t = sp.Symbol('t')
phi = sp.sin(6 * t) + math.pi/4
psi = sp.sin(t)

# speed and acceleration of point A
Vxa = sp.diff(sp.sin(phi) * R, t)
Vya = sp.diff(sp.cos(phi) * R, t)
Va = (Vxa**2 + Vya**2)**0.5
Wa = (sp.diff(Vxa, t)**2 + sp.diff(Vya, t)**2)**0.5


# speed and acceleration of point B
Vxb = sp.diff(sp.sin(psi) * BC, t)
Vyb = sp.diff(sp.cos(psi) * BC, t)
Vb = (Vxb**2 + Vyb**2)**0.5
Wb = (sp.diff(Vxb, t)**2 + sp.diff(Vyb, t)**2)**0.5

T = np.linspace(0, 2*math.pi, 1000)
XO = np.zeros_like(T)
YO = np.zeros_like(T)
XB = np.zeros_like(T)
YB = np.zeros_like(T)
VA = np.zeros_like(T)
WA = np.zeros_like(T)
VB = np.zeros_like(T)
WB = np.zeros_like(T)


for i in np.arange(len(T)):
    XO[i] = sp.Subs(R * sp.sin(phi), t, T[i])
    YO[i] = sp.Subs(R * sp.cos(phi), t, T[i])
    XB[i] = sp.Subs(BC * sp.cos(2 * psi), t, T[i])
    YB[i] = sp.Subs(-math.sqrt(BC**2 - XB[i]**2), t, T[i])
    VA[i] = sp.Subs(Va, t, T[i])
    WA[i] = sp.Subs(Wa, t, T[i])
    VB[i] = sp.Subs(Vb, t, T[i])
    WB[i] = sp.Subs(Wb, t, T[i])



fig = plt.figure(figsize=(8, 12))

ax1 = fig.add_subplot(1, 2, 1)
ax1.axis('equal')

Beam_AO, = ax1.plot([R, R -XO[0]], [0, -YO[0]], 'r')
P, = ax1.plot(R + OC, -R, 'black', marker='o')
Beam_OC, = ax1.plot([R, R + OC], [-R, -R], 'black')
Wall = Beam_CB, = ax1.plot([R + OC, R + OC], [-R, -R - BC], 'black')
Beam_CB, = ax1.plot([R + OC, R + OC + XB[0]], [-R, -R - YB[0]], 'g')
circleB, = ax1.plot(*CircleB(R + OC + XB[0] , -R - YB[0]), 'g')
circle, = ax1.plot(*Circle(R, 0), 'black')

anim = FuncAnimation(fig, anima, frames=1000, interval=0.01, blit=True)

plt.show()

