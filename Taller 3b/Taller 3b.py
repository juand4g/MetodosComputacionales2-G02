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

#PUNTO 3


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

