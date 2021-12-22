import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
from sympy import *
import math
from scipy.integrate import odeint


#  функция создания окружности
def circle(radius, center_x, center_y):
    coord_x = [center_x + radius * math.cos(i / 100) for i in range(0, 628)]
    coord_y = [center_y + radius * math.sin(i / 100) for i in range(0, 628)]
    return coord_x, coord_y


def odesys(y, t, fom1, fom2):
    y1, y2, y3, y4 = y
    dydt = [y3, y4, fom1(y1, y2, y3, y4), fom2(y1, y2, y3, y4)]
    return dydt


# параметры системы
m1 = 5
m2 = 1
l0 = 5
R = l0 * 0.5

AX = 0
AY = 0
Rc = 0.4
CX = l0
CY = - R - Rc

# коэффициенты
g = 9.81
alpha = 1

# определяем t как символ
t = sp.Symbol('t')

# определяем обобщенные координаты
phi = sp.Function('phi')(t)
psi = sp.Function('psi')(t)
omega_phi = sp.Function('omega_phi')(t)
omega_psi = sp.Function('omega_psi')(t)

# строим уравнения Лагранжа
# 1) определяем кинетическую энергию

# кинетическая энергия барабана
Ia = (m1 * R ** 2) / 2  # момент инерции барабана
TTa = (Ia * omega_phi ** 2) / 2  # кинетическая энергия

# кинетическая энергия груза
Vr = omega_phi * R
Ve = omega_psi * (l0 - phi * R)
Vb = Vr ** 2 + Ve ** 2
TTb = (m2 * Vb) / 2  # кинетическая энергия груза

TT = TTa + TTb  # общая кинетическая энергия

# 2) определяем потенциальную энергию
Pi = - m2 * g * (l0 - phi * R) * sp.cos(phi)

# 3) определяем непотенциальную энергию
M = alpha * phi

# функция Лагранжа
L = TT - Pi

# уравнения Лагранжа
ur1 = sp.diff(sp.diff(L, omega_phi), t) - sp.diff(L, phi) - M
ur2 = sp.diff(sp.diff(L, omega_psi), t) - sp.diff(L, psi)

massa1 = sp.Function('massa1')(t)
massa2 = sp.Function('massa2')(t)
length0 = sp.Function('length0')(t)
Radius = sp.Function('Radius')(t)
alpha_sym = sp.Function('alpha')(t)
g_sym = sp.Function('g')(t)
Ia_sym = (massa1 * Radius ** 2) / 2
TTa_sym = (Ia_sym * omega_phi ** 2) / 2
Vr_sym = omega_phi * Radius
Ve_sym = omega_psi * (length0 - phi * Radius)
Vb_sym = Vr_sym ** 2 + Ve_sym ** 2
TTb_sym = (massa2 * Vb_sym) / 2
TT_sym = TTa_sym + TTb_sym
Pi_sym = - massa2 * g_sym * (length0 - phi * Radius) * sp.cos(psi)
M_sym = alpha_sym * phi
L_sym = TT_sym - Pi_sym
ur1_sym = sp.diff(sp.diff(L_sym, omega_phi), t) - sp.diff(L_sym, phi) - M_sym
ur2_sym = sp.diff(sp.diff(L_sym, omega_psi), t) - sp.diff(L_sym, psi)

print(ur1_sym)
print(ur2_sym)

# выделяем вторые производные методом Крамера
a11 = ur1.coeff(sp.diff(omega_phi, t), 1)
a12 = ur1.coeff(sp.diff(omega_psi, t), 1)
a21 = ur2.coeff(sp.diff(omega_phi, t), 1)
a22 = ur2.coeff(sp.diff(omega_psi, t), 1)
b1 = -(ur1.coeff(sp.diff(omega_phi, t), 0)).coeff(sp.diff(omega_psi, t), 0).subs([(sp.diff(phi, t), omega_phi), (sp.diff(psi, t), omega_psi)])
b2 = -(ur2.coeff(sp.diff(omega_phi, t), 0)).coeff(sp.diff(omega_psi, t), 0).subs([(sp.diff(phi, t), omega_phi), (sp.diff(psi, t), omega_psi)])

detA = a11 * a22 - a12 * a21
detA1 = b1 * a22 - b2 * a21
detA2 = a11 * b2 - b1 * a21

DphiDt = detA1 / detA
DpsiDt = detA2 / detA

countOfFrames = 500

# создаем систему дифференциальных уравнений
T = np.linspace(0, 12, countOfFrames)
fom1 = sp.lambdify([phi, psi, omega_phi, omega_psi], DphiDt, "numpy")
fom2 = sp.lambdify([phi, psi, omega_phi, omega_psi], DpsiDt, "numpy")

# начальные значения
phi0 = 0
psi0 = 0.03
dphi0 = -0.01
dpsi0 = -0.01
y0 = [phi0, psi0, dphi0, dpsi0]

# получаем решение системы диффуров
# ПРОИЗВОДИМ ЧИСЛЕННОЕ ИНТЕГРИРОВАНИЕ
sol = odeint(odesys, y0, T, args=(fom1, fom2))

#sol - our solution
#sol[:,0] - phi
#sol[:,1] - psi
#sol[:,2] - dphi/dt
#sol[:,3] - dpsi/dt

# построение графика и подграфика с выравниманием осей

# левая часть - график с рисунком
fig = plt.figure(figsize=[22, 15])
ax = fig.add_subplot(1, 2, 1)
ax.axis('equal')
ax.set(xlim=[-10, 15], ylim=[-15, 10])
phi = sol[:, 0]
psi = sol[:, 1]
Vphi = sol[:, 2]
Vpsi = sol[:, 3]

# статичные объекты
Circ_A = ax.plot(*circle(R, AX, AY), 'black', linewidth=2)
Circ_C = ax.plot(*circle(Rc, CX, CY), 'black',  linewidth=2)
Line_AC = ax.plot([AX, CX], [AY - R, CY + Rc], 'black', linewidth=1)

# динамичные объекты
Rad_A,= ax.plot([AX, R * sp.sin(1.5 * phi[0])], [AY, - R * sp.cos(1.5 * phi[0])], 'black')
Line_CB, = ax.plot([AX + l0 + Rc, AX + l0 + Rc + l0 * sp.sin(1.5 * psi[0]) - l0 * sp.sin(1.5 * phi[0])], [AY - R - Rc, - Rc - l0 * sp.cos(1.5 * psi[0]) - l0 * sp.cos(1.5 * phi[0])], 'black', linewidth=1)
Circ_B, = ax.plot(AX + l0 + Rc + l0 * sp.sin(1.5 * psi[0]) - l0 * sp.sin(1.5 * phi[0]), - Rc - l0 * sp.cos(1.5 * psi[0]) - l0 * sp.cos(1.5 * phi[0]), marker='o', markersize=10, color='black')

# доп. графики
ax1 = fig.add_subplot(4, 2, 2)
ax1.plot(T, sol[:, 0])
ax1.set_title('phi(t)')
ax1.set_xlabel('T')
ax1.set_ylabel('phi(t)')

ax2 = fig.add_subplot(4, 2, 4)
ax2.plot(T, sol[:, 1])
ax1.set_title('psi(t)')
ax2.set_xlabel('T')
ax2.set_ylabel('psi(t)')

ax3 = fig.add_subplot(4, 2, 6)
ax3.plot(T, sol[:, 2])
ax3.set_title("phi(t)'")
ax3.set_xlabel('T')
ax3.set_ylabel("phi(t)'")

ax4 = fig.add_subplot(4, 2, 8)
ax4.plot(T, sol[:, 3])
ax4.set_title("psi(t)'")
ax4.set_xlabel('T')
ax4.set_ylabel("psi(t)'")

# расстояние между графиками
plt.subplots_adjust(wspace=0.3, hspace=0.7)


# функция анимации
def anima(i):
    Rad_A.set_data([AX, R * sp.sin(1.5 * phi[i])], [AY, - R * sp.cos(1.5 * phi[i])])
    Line_CB.set_data([AX + l0 + Rc, AX + l0 + Rc + l0 * sp.sin(1.5 * psi[i]) - l0 * sp.sin(1.5 * phi[i])], [AY - R - Rc, - Rc - l0 * sp.cos(1.5 * psi[i]) - l0 * sp.cos(1.5 * phi[i])])
    Circ_B.set_data(AX + l0 + Rc + l0 * sp.sin(1.5 * psi[i]) - l0 * sp.sin(1.5 * phi[i]), - Rc - l0 * sp.cos(1.5 * psi[i]) - l0 * sp.cos(1.5 * phi[i]))
    return Rad_A, Line_CB, Circ_B


# вызов функции анимации и демонстрация получившегося результата
anima = FuncAnimation(fig, anima, frames=countOfFrames, interval=100, blit=True)
plt.show()