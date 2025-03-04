import numpy as np
import matplotlib.pyplot as plt

# Parámetros físicos y numéricos
alpha = 0.022
x_min, x_max = 0, 2
Nx = 100  # Puntos espaciales
dx = (x_max - x_min) / (Nx - 1)
dt = 0.001  # Paso de tiempo
Nt = 1500  # Pasos de tiempo

# Malla espacial y temporal
x = np.linspace(x_min, x_max, Nx)
t = np.linspace(0, Nt * dt, Nt)

# Condiciones iniciales
U = np.cos(np.pi * x)

# Matriz para almacenar la evolución temporal
U_history = np.zeros((Nt, Nx))
U_history[0, :] = U

# Evolución temporal usando el esquema numérico
for n in range(1, Nt):
    U_next = U.copy()

    # Cálculo de derivadas con diferencias finitas
    dU_dx = (U[2:] - U[:-2]) / (2 * dx)
    d3U_dx3 = (U[2:] - 2 * U[1:-1] + U[:-2]) / dx**3

    # Aplicar el esquema de actualización
    U_next[1:-1] = U[1:-1] - dt * U[1:-1] * dU_dx - alpha * dt * d3U_dx3

    # Condiciones de frontera periódicas
    U_next[0] = U_next[-2]
    U_next[-1] = U_next[1]

    # Guardar el estado en la matriz
    U_history[n, :] = U_next
    U = U_next  # Avanzar en el tiempo

# Graficar el heatmap
plt.figure(figsize=(10, 3))
plt.imshow(U_history.T, aspect='auto', origin='lower',
           extent=[t.min(), t.max(), x.min(), x.max()],
           cmap="magma", interpolation="bilinear")

# Etiquetas y barra de color
plt.colorbar(label=r"$\psi(t, x)$")
plt.xlabel("Time [s]")
plt.ylabel("Angle x [m]")
plt.show()