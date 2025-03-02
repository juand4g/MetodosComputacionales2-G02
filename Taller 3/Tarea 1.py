import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def equations(t, y, beta, g):
    vx, vy, x, y_pos = y
    v = np.sqrt(vx ** 2 + vy ** 2)
    ax = -beta * v * vx
    ay = -g - beta * v * vy
    return [ax, ay, vx, vy]


def simulate_trajectory(theta, beta, v0=10, g=9.773):
    theta_rad = np.radians(theta)
    vx0 = v0 * np.cos(theta_rad)
    vy0 = v0 * np.sin(theta_rad)

    y0 = [vx0, vy0, 0, 0]
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)
    sol = solve_ivp(equations, t_span, y0, args=(beta, g), t_eval=t_eval, dense_output=True)

    x_vals = sol.y[2]
    y_vals = sol.y[3]

    mask = y_vals >= 0  # Consider only points where y >= 0
    return x_vals[mask], y_vals[mask]


def find_optimal_angle(beta):
    angles = np.linspace(0, 90, 50)
    max_range = 0
    best_angle = 0

    for theta in angles:
        x_vals, y_vals = simulate_trajectory(theta, beta)
        range_ = x_vals[-1]  # Last x-value before hitting the ground
        if range_ > max_range:
            max_range = range_
            best_angle = theta

    return best_angle, max_range


