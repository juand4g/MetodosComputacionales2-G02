import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Parámetros del problema
m = 10  # Masa de la bala (kg)
v0 = 10  # Velocidad inicial (m/s)
g = 9.773  # Gravedad en Bogotá (m/s^2)
beta_values = np.logspace(-3, 0, 100)  # Valores de beta en escala logarítmica


# Función para las ecuaciones diferenciales
def ecuaciones(y, t, beta, m):
    x, y_pos, vx, vy = y
    v = np.sqrt(vx ** 2 + vy ** 2)  # Velocidad total
    # Fuerzas de fricción
    F_friccion_x = -beta * vx * v ** 2
    F_friccion_y = -beta * vy * v ** 2
    # Ecuaciones de movimiento
    ax = F_friccion_x / m
    ay = (F_friccion_y / m) - g
    return [vx, vy, ax, ay]


# Función para simular el vuelo de la bala para un ángulo dado
def simular_vuelo(theta, beta):
    # Condiciones iniciales: en el origen, velocidad inicial en x y y
    y0 = [0, 0, v0 * np.cos(np.radians(theta)), v0 * np.sin(np.radians(theta))]
    # Tiempo de simulación
    t_max = 2 * v0 * np.sin(np.radians(theta)) / g
    t = np.linspace(0, t_max, 1000)

    # Resolver las ecuaciones diferenciales
    sol = odeint(ecuaciones, y0, t, args=(beta, m))

    # Obtener el alcance horizontal final (cuando y = 0)
    return sol[-1, 0], sol[-1, 2], sol[-1, 3]  # (alcance, vx_final, vy_final)


# Función para encontrar el ángulo óptimo de alcance máximo para un valor de beta
def obtener_angulo_maximo(beta):
    angulos = np.linspace(5, 80, 100)  # Probar ángulos de 5° a 80°
    alcances = []

    # Calcular el alcance para cada ángulo
    for theta in angulos:
        alcance, _, _ = simular_vuelo(theta, beta)
        alcances.append(alcance)

    # El ángulo que da el mayor alcance es el óptimo
    angulo_maximo = angulos[np.argmax(alcances)]
    return angulo_maximo


# Variables para almacenar los resultados
angulos_maximos = []
energia_perdidas = []

# Simulación para cada valor de beta
for beta in beta_values:
    # Encontrar el ángulo óptimo
    angulo_maximo = obtener_angulo_maximo(beta)
    # Simular el vuelo para el ángulo óptimo
    _, vx_final, vy_final = simular_vuelo(angulo_maximo, beta)

    # Energía cinética inicial
    energia_inicial = 0.5 * m * v0 ** 2

    # Energía cinética final
    energia_final = 0.5 * m * (vx_final ** 2 + vy_final ** 2)

    # Energía perdida por fricción
    energia_perdida = energia_inicial - energia_final

    angulos_maximos.append(angulo_maximo)
    energia_perdidas.append(energia_perdida)

# Gráfico del ángulo de alcance máximo vs coeficiente de fricción
plt.figure(figsize=(8, 6))
plt.plot(beta_values, angulos_maximos, label="Ángulo de alcance máximo")
plt.xscale('log')
plt.xlabel('Coeficiente de fricción β (kg/m)')
plt.ylabel('Ángulo de alcance máximo (°)')
plt.title('Ángulo de alcance máximo vs Coeficiente de fricción')
plt.grid(True)
plt.savefig('1.a.pdf')

# Gráfico de la energía perdida vs coeficiente de fricción
plt.figure(figsize=(8, 6))
plt.plot(beta_values, energia_perdidas, label="Energía perdida por fricción", color='red')
plt.xscale('log')
plt.xlabel('Coeficiente de fricción β (kg/m)')
plt.ylabel('Energía perdida (J)')
plt.title('Energía perdida vs Coeficiente de fricción')
plt.grid(True)
plt.savefig('1.b.pdf')


print('Angulo maximo: '+str(angulo_maximo))
