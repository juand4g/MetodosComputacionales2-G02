import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm  # Importar tqdm para la barra de progreso
import os
# Obtener la ruta absoluta del directorio donde está el archivo .py
script_dir = os.path.dirname(os.path.abspath(__file__))

#PUNTO 4 =======================================================================================================================================================
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

E_lowest = np.linspace(0,4.4,16000)
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
