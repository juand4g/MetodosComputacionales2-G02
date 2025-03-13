import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation

script_dir = os.path.dirname(os.path.abspath(__file__))

#PUNTO 1_A-------------------------------------------------------------------------------------------------------------------------------------------
def g_x(x, n=10, alpha=4/5):
    return sum(np.exp(-((x - k)**2) * k) / (k**alpha) for k in range(1,n))

def metropolis_hastings(g_x, num_samples=500000, x_init=0, proposal_width=1.0):
    samples = []
    x = x_init
    for _ in range(num_samples):
        x_new = x + np.random.uniform(-proposal_width, proposal_width)
        acceptance_ratio = g_x(x_new) / g_x(x) if g_x(x) > 0 else 0
        if np.random.rand() < acceptance_ratio:
            x = x_new
        samples.append(x)
    return np.array(samples)

# Generar datos aleatorios
samples = metropolis_hastings(g_x)

# Graficar y guardar el histograma
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=200, density=True, alpha=0.6, color='b')
plt.xlabel('x')
plt.ylabel('Densidad')
plt.title('Histograma de muestras generadas')
plt.savefig(os.path.join(script_dir,'1.a.pdf'))


#PUNTO 1_B--------------------------------------------------------------------------------------------------------------------------------------------
def f_x(x):
    return np.exp(-x**2)

sqrt_pi = np.sqrt(np.pi)
ratios = f_x(samples) / g_x(samples)
A = sqrt_pi / np.mean(ratios)
A_std = sqrt_pi * np.std(ratios) / (np.sqrt(len(samples)) * np.mean(ratios)**2)

print(f"1.b) A = {A:.3f} ± {A_std:.3f}")

#PUNTO 3 -----------------------------------------------------------------------------------------------------------------------------------------------
# Parámetros del sistema
N = 150
J = 0.2
beta = 10
frames = 500
iterations_per_frame = 400

# Inicializar la malla de espines aleatoriamente
spins = np.random.choice([-1, 1], size=(N, N))

# Función para calcular la energía de un solo espín
def calculate_energy_change(spins, i, j):
    top = spins[(i-1)%N, j]
    bottom = spins[(i+1)%N, j]
    left = spins[i, (j-1)%N]
    right = spins[i, (j+1)%N]
    return 2 * J * spins[i, j] * (top + bottom + left + right)

# Función para realizar una iteración del algoritmo de Metrópolis
def metropolis_step(spins):
    for _ in range(iterations_per_frame):
        i, j = np.random.randint(0, N, 2)
        delta_E = calculate_energy_change(spins, i, j)
        if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E):
            spins[i, j] *= -1
    return spins

# Configuración de la animación
fig, ax = plt.subplots()
im = ax.imshow(spins, cmap='gray', vmin=-1, vmax=1)
ax.set_xticks([])
ax.set_yticks([])

def update(frame):
    global spins
    spins = metropolis_step(spins)
    im.set_data(spins)
    return [im]

# Crear la animación
ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Guardar la animación en un archivo de video
ani.save(os.path.join(script_dir, '3.mp4'), writer='ffmpeg')

