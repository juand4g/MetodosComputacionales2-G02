import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
from tqdm import tqdm  # Importar tqdm para la barra de progreso
import os
# Obtener la ruta absoluta del directorio donde está el archivo .py
script_dir = os.path.dirname(os.path.abspath(__file__))


#PUNTO 1

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

















#PUNTO 3

# Constantes
mu = 39.4234021  #Parámetro gravitacional en AU^3/Year^2

# Parámetros orbitales de Mercurio
a = 0.38709893  # Semieje mayor en UA
e = 0.20563069  # Excentricidad

# Condiciones inicial
x0 = a * (1 + e)  # x(t=0)
y0 = 0.0         # y(t=0)

# Velocidad inicial
v_y0 = np.sqrt(mu / a) * np.sqrt((1 - e) / (1 + e))
v_x0 = 0.0

# Tiempo de simulación
t_span = (0, 10)  # Simulación durante 10 años
t_paso = np.linspace(t_span[0], t_span[1], int(2e3))  # Resolución temporal

def EDO(t, state):
    """
    Define las ecuaciones diferenciales del sistema.
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r = np.sqrt(x**2 + y**2)  # Distancia al Sol
    r_dir = np.array([x, y]) / r  # Vector unitario radial

    # Fuerza gravitacional con corrección relativista
    a = -(mu / r**2) * (1 + alpha / r**2)

    # Aceleración
    ax = a * r_dir[0]
    ay = a * r_dir[1]

    return [vx, vy, ax, ay]
# Condiciones iniciales
c_iniciales = [x0, y0, v_x0, v_y0]

# Resolver las ecuaciones diferenciales
def eventos(t, state):
    x, y, vx, vy = state
    
    return x*vx+y*vy
eventos.direction = 1

#Parte a)
alpha = 1e-2
solucion = solve_ivp(
    EDO,
    t_span=t_span,
    y0=c_iniciales,
    t_eval=t_paso,
    method="Radau",
    max_step = 1e-3
)
x = solucion.y[0]
y = solucion.y[1]
vx = solucion.y[2]
vy = solucion.y[3]
t = solucion.t

# Crear la figura y los ejes
fig, ax = plt.subplots()
plt.title("Precesión de Mercurio")
plt.scatter([0], [0], color='yellow', label="Sol")  # Marcar el Sol
ax.set_xlim(-0.5, 0.5)  # Límites del eje x
ax.set_ylim(-0.5, 0.5)  # Límites del eje y
ax.set_aspect('equal')  # Aspecto igual para x e y

# Crear la línea para la trayectoria
line, = ax.plot([], [], lw=2, label="Órbita de Mercurio")

# Crear el punto para la posición actual
point, = ax.plot([], [], 'ro', label="Mercurio")

# Añadir leyenda
plt.legend(loc="upper left")

# Función de inicialización
def init():
    line.set_data([], [])  # Línea vacía
    point.set_data([], [])  # Punto vacío
    return line, point

# Función de actualización
def animate(i):
    # Actualizar la línea con los datos hasta el cuadro i
    line.set_data(x[:i], y[:i])

    # Actualizar el punto con la posición actual
    point.set_data([x[i-1]], [y[i-1]])

    return line, point

# Crear la animación
ani = FuncAnimation(
    fig,                # Figura
    animate,            # Función de actualización
    frames=len(t),      # Número de cuadros
    init_func=init,     # Función de inicialización
    blit=True,           # Optimización

)

# Guardar la animación como un archivo mp4
ani.save("3.a.mp4")

#Parte b)
alpha = 1.09778201e-8
solucion = solve_ivp(
    EDO,
    t_span=t_span,
    y0=c_iniciales,
    t_eval=t_paso,
    method="Radau",
    max_step = 1e-3,
    events=eventos
)

# Extraer resultados
x = solucion.y[0]
y = solucion.y[1]
vx = solucion.y[2]
vy = solucion.y[3]
t = solucion.t
ev_t = solucion.t_events[0]
ev = solucion.y_events[0]
ev_x = ev[:,0]
ev_y = ev[:,1]

#Hallar los ángulos en arcsec
theta = np.arctan2(ev_y, ev_x)
theta_grad =np.rad2deg(theta)+180
theta_arcs = theta_grad*3600

#Realizar le ajuste
(m,b), cov = np.polyfit(ev_t,theta_arcs,1,cov=True)
m_err = np.sqrt(cov[0, 0]) 
b_err = np.sqrt(cov[1, 1])  

# Graficar
plt.figure(figsize=(8, 6))
plt.scatter(ev_t, theta_arcs, label="Datos", color="C0")
plt.plot(t_peri, m * t_peri + b, "--", label=f"Ajuste lineal: {m:.7f} ± {m_err:.7f} arcsec/año", color="orange")
plt.xlabel("Tiempo (años)")
plt.ylabel("Ángulo del periastro (arcsec)")
plt.title("Precesión del periastro de Mercurio")
plt.legend()
plt.savefig("3.b.pdf")

print(f" Se calculó una precesión para el periastro de {m:.7f} ± {m_err:.7f} arcsec/año. Así, basado en la literatura de una precesión de 0.429799 arc/año, se puede ver que hay un error porcentual de tan sólo 0.4%")

#PUNTO 4 