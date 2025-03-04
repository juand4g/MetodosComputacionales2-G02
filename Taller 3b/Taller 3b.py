import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numba import njit
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import matplotlib.animation as animation
# Obtener la ruta absoluta del directorio donde está el archivo .py
script_dir = os.path.dirname(os.path.abspath(__file__))

#PUNTO 1
# Parámetros de la malla
N = 100  # Número de puntos en cada dirección
h = 2.0 / (N - 1)  # Tamaño de celda (cubre [-1.1, 1.1])

# Crear la malla
x = np.linspace(-1.1, 1.1, N)
y = np.linspace(-1.1, 1.1, N)
X, Y = np.meshgrid(x, y)

# Función de densidad de carga
rho = -X - Y

# Inicializar phi con valores aleatorios en el interior del disco
phi = np.random.rand(N, N) * 0.1

# Aplicar condiciones de frontera
R = np.sqrt(X**2 + Y**2)
Theta = np.arctan2(Y, X)
phi[R >= 1] = np.sin(7 * Theta[R >= 1])

# Iteraciones del método de relajación
max_iter = 15000
tol = 1e-4

@njit
def relajacion(phi, rho, R, h, max_iter, tol, N):
    for k in range(max_iter):
        phi_old = phi.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                if R[i, j] < 1:  # Solo actualizamos dentro del disco
                    phi[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1] - h**2 * (-4 * np.pi * rho[i, j]))
        if np.sum(np.abs(phi - phi_old)) < tol:
            print(f'Convergencia alcanzada en {k+1} iteraciones')
            break
    return phi

# Ejecutar el método de relajación
phi = relajacion(phi, rho, R, h, max_iter, tol, N)

# Crear la figura y el subplot 3D
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(121, projection='3d')

# Usar un colormap que vaya de violeta a rojo
cmap = cm.get_cmap("plasma")
norm = mcolors.Normalize(vmin=np.min(phi), vmax=np.max(phi))

# Graficar la superficie
surf = ax.plot_surface(X, Y, phi, cmap=cmap, norm=norm)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Φ(X, Y)")
ax.set_title("Solución de la Ecuación de Poisson")
plt.savefig(os.path.join(script_dir,'1.png'))

#PUNTO 2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
# Asegurar que se use el backend correcto
plt.rcParams["animation.html"] = "jshtml"

# Parámetros del problema
L = 2  # Longitud del dominio
Nx = 100  # Número de puntos espaciales
dx = L / (Nx - 1)  # Paso espacial
c = 1  # Velocidad de la onda
dt = 0.5 * dx / c  # Paso temporal (satisfaciendo la condición de Courant)
Nt = 200  # Número de pasos temporales

# Malla espacial
x = np.linspace(0, L, Nx)

# Condición inicial: un pulso gaussiano
u0 = np.exp(-125 * (x - L/2)**2)

# Inicializar listas para cada condición de frontera
conds_frontera = ["Dirichlet", "Neumann", "Periódicas"]
u_vals = [np.copy(u0) for _ in conds_frontera]
u_prev_vals = [np.copy(u0) for _ in conds_frontera]

# Función para actualizar la onda en el tiempo
def update_wave(u, u_prev, bc_type):
    u_new = np.zeros_like(u)
    for i in range(1, Nx - 1):
        u_new[i] = 2 * u[i] - u_prev[i] + (c * dt / dx) ** 2 * (u[i+1] - 2 * u[i] + u[i-1])
    
    # Aplicar condiciones de frontera
    if bc_type == "Dirichlet":
        u_new[0] = u_new[-1] = 0
    elif bc_type == "Neumann":
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]
    elif bc_type == "Periódicas":
        u_new[0] = u_prev[-2]  # Condición periódica
        u_new[-1] = u_prev[1]  # Condición periódica
    
    return u_new, u

# Inicializar la figura
fig, axes = plt.subplots(3, 1, figsize=(6, 8))
lines = []

for ax, cond in zip(axes, conds_frontera):
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.set_title(f"Condición de frontera: {cond}")
    line, = ax.plot(x, u0, lw=2)
    lines.append(line)

# Función de animación
def animate(frame):
    global u_vals, u_prev_vals
    for i, cond in enumerate(conds_frontera):
        u_vals[i], u_prev_vals[i] = update_wave(u_vals[i], u_prev_vals[i], cond)
        lines[i].set_ydata(u_vals[i])
    return lines

# Crear la animación (sin blit para compatibilidad con notebook)
ani = animation.FuncAnimation(fig, animate, frames=Nt, interval=30, blit=False)

HTML(ani.to_jshtml())

# Guardar la animación en un archivo mp4 DESPUÉS de mostrarla
ani.save("2.mp4", writer="ffmpeg", fps=30)

#PUNTO 3
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

# PUNTO 4

# Parámetros del problema
Lx, Ly = 2.0, 1.0  # Dimensiones del tanque (m)
dx, dy = 0.01, 0.01  # Resolución espacial (m)
dt = 0.002  # Paso temporal (s)

c_base = 0.5  # Velocidad base de la onda (m/s)
c_lente = c_base / 5  # Velocidad dentro de la lente (m/s)

# Definir la grilla
nx, ny = int(Lx / dx), int(Ly / dy)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Inicializar el campo de velocidades
c = np.full((ny, nx), c_base)

# Definir la lente como una media elipse
def es_lente(x, y):
    return ((x - Lx/4)**2 + 3*(y - Ly/2)**2 / 25) <= 1/25 and y > Ly/2

for i in range(ny):
    for j in range(nx):
        if es_lente(x[j], y[i]):
            c[i, j] = c_lente

# Condiciones iniciales
u = np.zeros((ny, nx))
u_prev = np.zeros((ny, nx))
u_next = np.zeros((ny, nx))

# Fuente de onda
source_x, source_y = int(0.5 / dx), int(0.5 / dy)
freq = 10  # Frecuencia (Hz)

# Parámetro de estabilidad de Courant
CFL = np.max(c) * dt / dx
assert CFL < 1, "El coeficiente de Courant debe ser menor que 1 para estabilidad"

# Función de actualización de la onda
def update(frame):
    global u, u_prev, u_next
    
    # Aplicar ecuación de onda con diferencias finitas
    for i in range(1, ny-1):
        for j in range(1, nx-1):
            c2 = c[i, j]**2
            u_next[i, j] = (2*u[i, j] - u_prev[i, j] + 
                            (dt**2 / dx**2) * c2 * (u[i+1, j] - 2*u[i, j] + u[i-1, j]) +
                            (dt**2 / dy**2) * c2 * (u[i, j+1] - 2*u[i, j] + u[i, j-1]))
    
    # Aplicar la fuente
    u_next[source_y, source_x] += np.sin(2 * np.pi * freq * frame * dt)
    
    # Intercambiar buffers
    u_prev, u, u_next = u, u_next, u_prev
    
    # Actualizar gráfico
    im.set_array(u)
    return [im]

# Crear la animación
fig, ax = plt.subplots()
im = ax.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='bwr', vmin=-1, vmax=1)
ax.set_title("Simulación de la ecuación de onda 2D")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ani = animation.FuncAnimation(fig, update, frames=1000, interval=20, blit=True)

# Guardar el video
ani.save(os.path.join(script_dir,"4_a.mp4"), writer="ffmpeg", fps=50)

