import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
archivo = open('C:/Users/Usuario/Desktop/H_field.csv', 'r')
data = pd.read_csv(archivo)

# Suponiendo que las columnas sean 't' y 'H'
t = data["t"].values
H = data["H"].values

# Calcular la transformada rápida de Fourier con np.fft.fft()
n = len(t)
dt = np.mean(np.diff(t))  # Paso de tiempo promedio

H_fft = np.fft.fft(H)
freqs = np.fft.fftfreq(n, dt)

# Obtener la frecuencia dominante f_general
f_general = freqs[np.argmax(np.abs(H_fft))]

# Suponiendo que f_fast ya fue calculado anteriormente
f_fast = freqs[np.argmax(np.abs(np.fft.rfft(H)))]  # Esto también se puede calcular con rfft si prefieres

# Imprimir los valores
print(f"2.a) {f_fast = :.5f}; {f_general = :.5f}")

# Calcular las fases
phi_fast = np.mod(f_fast * t, 1)
phi_general = np.mod(f_general * t, 1)

# Graficar
plt.figure(figsize=(8, 6))
plt.scatter(phi_fast, H, label="H vs φ_fast", alpha=0.5)
plt.scatter(phi_general, H, label="H vs φ_general", alpha=0.5)
plt.xlabel("Fase φ")
plt.ylabel("Campo H")
plt.legend()
plt.title("Comparación de Fases con f_fast y f_general")
plt.savefig("2.a.pdf")
plt.show()