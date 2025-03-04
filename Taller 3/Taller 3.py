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
plt.savefig(os.path.join(script_dir,'1.a.pdf'))

# Gráfico de la energía perdida vs coeficiente de fricción
plt.figure(figsize=(8, 6))
plt.plot(beta_values, energia_perdidas, label="Energía perdida por fricción", color='red')
plt.xscale('log')
plt.xlabel('Coeficiente de fricción β (kg/m)')
plt.ylabel('Energía perdida (J)')
plt.title('Energía perdida vs Coeficiente de fricción')
plt.grid(True)
plt.savefig(os.path.join(script_dir,'1.b.pdf'))


print('Angulo maximo: '+str(angulo_maximo))


#   PUNTO 2 














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
plt.plot(ev_t, m * ev_t + b, "--", label=f"Ajuste lineal: {m:.7f} ± {m_err:.7f} arcsec/año", color="orange")
plt.xlabel("Tiempo (años)")
plt.ylabel("Ángulo del periastro (arcsec)")
plt.title("Precesión del periastro de Mercurio")
plt.legend()
plt.savefig("3.b.pdf")

print(f" Se calculó una precesión para el periastro de {m:.7f} ± {m_err:.7f} arcsec/año. Así, basado en la literatura de una precesión de 0.429799 arc/año, se puede ver que hay un error porcentual de tan sólo 0.4%")

#PUNTO 4 
def schrodinger(x, y, E):
    """Sistema de ecuaciones de Schrödinger para el oscilador armónico."""
    f, df_dx = y
    return [df_dx, (x**2 - 2*E) * f]

def solve_shooting(E, symmetric=True, x_max=6, x_steps=1000, metodo="RK45"):
    """Resuelve la ecuación con el método de disparo para un valor dado de E."""
    x_eval = np.linspace(0, x_max, x_steps)
    y0 = [1, 0] if symmetric else [0, 1]  # Condiciones iniciales
    sol = solve_ivp(schrodinger, [0, x_max], y0, args=(E,), t_eval=x_eval, method=metodo)
    return sol.t, sol.y[0], sol.y[1]

def find_eigenvalues(E_range, symmetric=True, tol=1.5, energy_threshold=0.3, met="RK45"):
    """Encuentra los valores de E para los cuales la solución no diverge evitando valores muy cercanos."""
    eigenvalues = []
    # Usar tqdm para mostrar la barra de progreso
    for E in tqdm(E_range, desc=f"Buscando energías permitidas. Método: {met}. Rango: ({E_range[0]:.1f}, {E_range[-1]:.1f})", unit="E"):
        x, f, df_dx = solve_shooting(E, symmetric, metodo=met)
        if np.abs(f[-1]) < tol and np.abs(df_dx[-1]) < tol:  # Condición de convergencia para f y su derivada
            if not eigenvalues or np.min(np.abs(np.array(eigenvalues) - E)) > energy_threshold:
                eigenvalues.append(E)
    return eigenvalues

# Rango de valores de E

E_lowest = np.linspace(0,4,8000)
E_low = np.arange(4, 5, 5/30001)
E_high = np.linspace(5, 10, 10000)

E_values = np.concatenate((E_lowest, E_low, E_high))



# Encontrar energías permitidas
print("Buscando energías simétricas...")
symmetric_eigenvalues_lowest = find_eigenvalues(E_lowest, symmetric=True)
symmetric_eigenvalues_low = find_eigenvalues(E_low, symmetric=True)
symmetric_eigenvalues_high = find_eigenvalues(E_high, symmetric=True)
symmetric_eigenvalues = symmetric_eigenvalues_lowest+symmetric_eigenvalues_low + symmetric_eigenvalues_high

#symmetric_eigenvalues = find_eigenvalues(E_values, symmetric=True, met="RK23")

print("Buscando energías antisimétricas...")
antisymmetric_eigenvalues_lowest = find_eigenvalues(E_lowest, symmetric=False)
antisymmetric_eigenvalues_low = find_eigenvalues(E_low, symmetric=False)
antisymmetric_eigenvalues_high = find_eigenvalues(E_high, symmetric=False)
antisymmetric_eigenvalues = antisymmetric_eigenvalues_lowest+antisymmetric_eigenvalues_low + antisymmetric_eigenvalues_high
#antisymmetric_eigenvalues = find_eigenvalues(E_values, symmetric=False, met="DOP853")

print("Energías permitidas (simétricas):", symmetric_eigenvalues)
print("Energías permitidas (antisimétricas):", antisymmetric_eigenvalues)

# Graficar algunas soluciones
plt.figure(figsize=(8, 5))
for E in symmetric_eigenvalues[:5]:
    x, f, _ = solve_shooting(E, symmetric=True)
    x_full = np.concatenate((-x[::-1], x))  # Reflejar x
    f_full__ = np.concatenate((f[::-1], f))   # Reflejar f
    f_full = f_full__/f_full__.max()
    plt.plot(x_full, f_full + E, label=f"E = {E:.2f} (simétrica)")
    plt.axhline(E, color="gray", alpha=0.3)
for E in antisymmetric_eigenvalues[:5]:
    x, f, _ = solve_shooting(E, symmetric=False)
    x_full = np.concatenate((-x[::-1], x))  # Reflejar x
    f_full__ = np.concatenate((-f[::-1], f))  # Reflejar f con cambio de signo
    f_full = f_full__/f_full__.max()
    plt.plot(x_full, f_full + E, '--', label=f"E = {E:.2f} (antisimétrica)")
    plt.axhline(E, color="gray", alpha=0.3)
plt.axhline(0, color='black', linewidth=0.5, linestyle='dashed')
plt.xlabel("x")
plt.ylabel("Energía")
plt.title("Energías permitidas y funciones de onda")
plt.legend()
plt.savefig(os.path.join(script_dir,"4.pdf"), dpi=300)
