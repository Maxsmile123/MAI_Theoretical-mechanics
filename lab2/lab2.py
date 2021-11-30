import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp
import math


# функция создания окружности
def Circle(X, Y, R):
    CX = [X + R * math.cos(i/100) for i in range(0, 628)]
    CY = [Y + R * math.sin(i/100) for i in range(0, 628)]
    return CX, CY


Steps = 1001
t = np.linspace(0, 10, Steps)  # массив времени
phi = math.pi/2 * np.sin(-2.1 * t)
psi = np.sin(2.1 * t)

time = sp.Symbol('t')
phi_copy = math.pi / 2 * sp.sin(-2.1 * time)
psi_copy = 2.1 * time

# Барабан. Точка А - центр барабана
A_R = 1 # Радиус барабана
A_X = 0 # Центр барабана по оси Х
A_Y = 0 # Центр барабана по оси Y
A_R_End_X = A_X + A_R * np.sin(phi)
A_R_End_Y = A_Y - A_R * np.cos(phi)

# трос
Length_0 = A_R * 3  # свешивающаяся часть
Length_1 = Length_0  # прямая часть

# Блок. Точка С - центр блока
C_R = 0.4
C_X = Length_1
C_Y = A_Y - A_R - C_R
C_R_End_X = C_X + C_R * np.sin(math.pi / 2 - psi) + 0.018
C_R_End_Y = C_Y + C_R * np.cos(math.pi / 2 - psi) - 0.0019

# груз
B_X = C_X + Length_0 * np.sin(psi)
B_Y = C_Y - Length_0 * np.cos(psi)

# левая часть - график с рисунком
fig = plt.figure(figsize=[17, 10])
ax = fig.add_subplot(1, 2, 1)
ax.axis('equal')
ax.set(xlim=[-5, 10], ylim=[-10, 5])

# создание рисунка
Line_AC = ax.plot([A_X, C_X], [A_Y - A_R, C_Y + C_R], 'black', linewidth=1)
Line_CB = ax.plot([C_X + C_R, B_X[0]], [C_Y, B_Y[0]], 'black', linewidth=1)[0]
Circle_A = ax.plot(*Circle(A_X, A_Y, A_R), 'black', linewidth=2)
Circle_C = ax.plot(*Circle(C_X, C_Y, C_R), 'black', linewidth=2)
Point_B = ax.plot(B_X[0], B_Y[0], marker='o', markersize=15, color='black')[0]
Line_A_Radius = ax.plot([A_X, A_R_End_X[0]], [A_Y, A_R_End_Y[0]], 'black')[0]


A_Length_Of_Sector = (math.pi * A_R / 180) * 0.95 * 2
Counter = A_R


# анимация системы
def anima(i, counter, a_sector):
    if i < Steps - 1:
        if B_X[i] == 4:  # состояние равновесия
            counter = 0
        elif B_X[i] < 4:  # груз справа от точки равновесия
            if B_Y[i] > B_Y[i + 1]:  # летит вверх
                counter -= a_sector
            else:  # летит вниз
                counter += a_sector
        else:  # груз слева от точки равновесия
            if B_Y[i] < B_Y[i + 1]:  # летит вверх
                counter += a_sector
            else:  # летит вниз
                counter -= a_sector
    Line_CB.set_data([C_R_End_X[i], B_X[i] - counter], [C_R_End_Y[i], B_Y[i] - counter])
    Point_B.set_data(B_X[i] - counter, B_Y[i] - counter)
    Line_A_Radius.set_data([A_X, A_R_End_X[i]], [A_Y, A_R_End_Y[i]])
    return [Point_B, Line_CB, Line_A_Radius]


# скорость и ускорение точки А
A_vx = sp.diff(sp.sin(phi_copy) * A_R, time)
A_vy = sp.diff(sp.cos(phi_copy) * A_R, time)
A_v = (A_vx ** 2 + A_vy ** 2) ** 0.5
A_w = (sp.diff(A_vx, time) ** 2 + sp.diff(A_vy, time) ** 2) ** 0.5

# скорость и ускорение точки В
B_vx = sp.diff(sp.sin(psi_copy) * Length_0, time) + A_vx
B_vy = sp.diff(sp.cos(psi_copy) * Length_0, time) + A_vy
b_v = (B_vx ** 2 + B_vy ** 2) ** 0.5
B_w = (sp.diff(B_vx, time) ** 2 + sp.diff(B_vy, time) ** 2) ** 0.5

# рисование графиков скоростей и ускорений
T = np.linspace(0, 2 * math.pi, 1000)
A_V = np.zeros_like(T)
A_W = np.zeros_like(T)
B_V = np.zeros_like(T)
B_W = np.zeros_like(T)

for i in np.arange(len(T)):
    A_V[i] = sp.Subs(A_v, time, T[i])
    A_W[i] = sp.Subs(A_w, time, T[i])
    B_V[i] = sp.Subs(b_v, time, T[i])
    B_W[i] = sp.Subs(B_w, time, T[i])

# скорость точки A
ax_av = fig.add_subplot(4, 2, 2)
ax_av.plot(T, A_V)
plt.title('speed of the point A')
plt.xlabel('T')
plt.ylabel('v of point A')

# ускорение точки A
ax_aw = fig.add_subplot(4, 2, 4)
ax_aw.plot(T, A_W)
plt.title('acceleration of the point A')
plt.xlabel('T')
plt.ylabel('w of point A')

# скорость точки B
ax_bv = fig.add_subplot(4, 2, 6)
ax_bv.plot(T, B_V)
plt.title('speed of the Point B')
plt.xlabel('T')
plt.ylabel('v of point B')

# ускорение точки B
ax_bw = fig.add_subplot(4, 2, 8)
ax_bw.plot(T, B_W)
plt.title('acceleration of the Point B')
plt.xlabel('T')
plt.ylabel('w of point B')

# расстояние между графиками скоростей и ускорений
plt.subplots_adjust(wspace=0.3, hspace=0.7)

# вызов функции анимации и демонстрация получившегося результата
anima = FuncAnimation(fig, anima, frames=Steps, fargs=(Counter, A_Length_Of_Sector,), interval=0.01)
plt.show()