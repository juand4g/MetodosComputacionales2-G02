# -*- coding: utf-8 -*-
"""3a2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cF6ddeTbuXQkQKGasmw-NaEmmCprFEDE
"""

import numpy as np
import matplotlib.pyplot as plt

# Parámetros de la simulación
dt = 0.001  # Paso de tiempo en unidades atómicas
num_steps = 10000  # Número de pasos de integración

# Condiciones iniciales
x = 1.0  # Radio inicial en unidades de Bohr
y = 0.0
vx = 0.0  # Velocidad inicial en x
vy = 1.0  # Velocidad inicial en y

# Listas para almacenar datos
x_vals, y_vals = [], []
r_vals, E_vals, t_vals = [], [], []

# Función para calcular la aceleración (Ley de Coulomb)
def acceleration(x, y):
    r = np.sqrt(x**2 + y**2)
    ax = -x / r**3
    ay = -y / r**3
    return ax, ay

# Inicializar simulación
t = 0
x_vals.append(x)
y_vals.append(y)
r_vals.append(np.sqrt(x**2 + y**2))
E_vals.append(0.5 * (vx**2 + vy**2) - 1/np.sqrt(x**2 + y**2))
t_vals.append(t)

# Primer paso con Verlet
ax, ay = acceleration(x, y)
x_new = x + vx * dt + 0.5 * ax * dt**2
y_new = y + vy * dt + 0.5 * ay * dt**2
vx += 0.5 * ax * dt
vy += 0.5 * ay * dt

# Integración con Verlet
periodo_simulado = None
for step in range(1, num_steps):
    x_old, y_old = x, y
    x, y = x_new, y_new

    x_vals.append(x)
    y_vals.append(y)

    ax, ay = acceleration(x, y)

    # Paso siguiente con Verlet
    x_new = 2 * x - x_old + ax * dt**2
    y_new = 2 * y - y_old + ay * dt**2

    vx += ax * dt
    vy += ay * dt

    # Guardar radio y energía
    r = np.sqrt(x**2 + y**2)
    E = 0.5 * (vx**2 + vy**2) - 1/r
    r_vals.append(r)
    E_vals.append(E)
    t_vals.append(t)

    # Detección del período (cuando y cruza de negativo a positivo)
    if step > 1 and y_old < 0 and y >= 0:
        periodo_simulado = t
        break

    t += dt

# Cálculo del período teórico
T_teorico = 2 * np.pi * 1**(3/2)  # Para a = 1 en unidades atómicas

# Imprimir resultados
print(f'2.a) P_teo = {T_teorico:.5f}; P_sim = {periodo_simulado:.5f}')

# Graficar la órbita
plt.figure(figsize=(6,6))
plt.plot(x_vals, y_vals, label="Órbita simulada")
plt.scatter([0], [0], color="red", label="Protón (núcleo)")
plt.xlabel("x (unidades de a0)")
plt.ylabel("y (unidades de a0)")
plt.title("Órbita del electrón en el potencial de Coulomb")
plt.legend()
plt.grid()
plt.axis("equal")
plt.show()

# Graficar el radio en función del tiempo
plt.figure(figsize=(6,4))
plt.plot(t_vals, r_vals, label="Radio $r(t)$", color="blue")
plt.xlabel("Tiempo (unidades atómicas)")
plt.ylabel("Radio $r$")
plt.title("Conservación del radio")
plt.legend()
plt.grid()
plt.show()

# Graficar la energía en función del tiempo
plt.figure(figsize=(6,4))
plt.plot(t_vals, E_vals, label="Energía total $E(t)$", color="green")
plt.xlabel("Tiempo (unidades atómicas)")
plt.ylabel("Energía total $E$")
plt.title("Conservación de la energía")
plt.legend()
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Constantes
q = 1  # Carga del electrón (en unidades reducidas)
m = 1  # Masa del electrón (en unidades reducidas)
dt = 0.001  # Paso de tiempo
t_max = 15  # Tiempo máximo de simulación
num_steps = int(t_max / dt)  # Número de pasos
alpha=1/137.035999206

# Condiciones iniciales
x, y = 1.0, 0.0  # Posición inicial (radio de Bohr)
vx, vy = 0.0, 1.0  # Velocidad inicial (órbita circular)

def derivatives(state):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax = -x / r**3
    ay = -y / r**3
    return np.array([vx, vy, ax, ay])
#def RK4(F,y0,t,dt):
    #k1 = F(t,y0)
    #k2 = F( t+dt/2, y0 + dt*k1/2 )
    #k3 = F( t+dt/2, y0 + dt*k2/2  )
    #k4 = F( t+dt, y0 + dt*k3  )
    #return y0 + dt/6 * (k1+2*k2+2*k3+k4)
def RK4(F, y0, dt):
    k1 = F(y0)
    k2 = F(y0 + dt*k1/2)
    k3 = F(y0 + dt*k2/2)
    k4 = F(y0 + dt*k3)
    return y0 + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

positions = []
energies = []
times = []
t = 0
state = np.array([x, y, vx, vy])

for _ in range(num_steps):
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)
    ax=derivatives(state)[2]
    ay=derivatives(state)[3]
    #state = RK4(derivatives,state,t,dt)

    state = RK4(derivatives, state, dt)
    v2 = vx**2 + vy**2
    v2_new = state[2]**2 + state[3]**2
    a_new= ax**2 + ay**2
    energy = 0.5 * v2
    positions.append((x, y))
    energies.append(energy)
    times.append(t)

    if r < 0.01:
        break  # Detener si el electrón cae al núcleo

    # Cálculo de pérdida de energía
    factorL=np.sqrt((v2) - (4/3)*a_new*alpha**3*dt)
    v_new = factorL
    #if v_new < 0:
    #    v_new = 0
    state[2] = v_new * state[2] / np.sqrt(v2_new) #if v2_new > 0 else 0
    state[3] = v_new * state[3] / np.sqrt(v2_new) #if v2_new > 0 else 0

    t += dt

time_fall = t
print(f"Tiempo de caída del electrón: {time_fall:.5f} as")

# Graficar trayectoria
positions = np.array(positions)
plt.figure(figsize=(6,6))
plt.plot(positions[:,0], positions[:,1], label="Órbita del electrón")
plt.scatter([0], [0], color='red', label="Núcleo")
plt.xlabel("x (unidades de Bohr)")
plt.ylabel("y (unidades de Bohr)")
plt.title("Trayectoria del electrón con radiación de Larmor")
plt.legend()
plt.grid()
plt.savefig("2.b.XY.pdf")

# Graficar energía en función del tiempo
plt.figure(figsize=(6,4))
plt.plot(times, energies, label="Energía total")
plt.xlabel("Tiempo")
plt.ylabel("Energía")
plt.title("Evolución de la energía del electrón")
plt.legend()
plt.grid()
plt.savefig("2.b.diagnostics.pdf")
plt.show()







