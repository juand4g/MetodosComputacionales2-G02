import numpy as np
import matplotlib.pyplot as plt


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
plt.savefig('1.a.pdf')


#PUNTO 1_B--------------------------------------------------------------------------------------------------------------------------------------------
def f_x(x):
    return np.exp(-x**2)

sqrt_pi = np.sqrt(np.pi)
ratios = f_x(samples) / g_x(samples)
A = sqrt_pi / np.mean(ratios)
A_std = sqrt_pi * np.std(ratios) / (np.sqrt(len(samples)) * np.mean(ratios)**2)

print(f"1.b) A = {A:.3f} ± {A_std:.3f}")