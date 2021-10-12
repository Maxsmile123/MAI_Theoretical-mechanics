import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import sympy as sp
import math

def Circle(X, Y):
    CX = [X + 0.75 * math.cos(i/100) for i in range(0, 628)]
    CY = [Y + 0.75 * math.sin(i/100) for i in range(0, 628)]
    return CX, CY

OC = 4
CB = 2

t = sp.Symbol('t')
phi = 3 * t
alpha = t    # angle between Oy and CB




