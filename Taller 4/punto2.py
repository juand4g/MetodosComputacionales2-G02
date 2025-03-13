# -*- coding: utf-8 -*-
"""punto2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YWB87eYJYrXKAfFbuzCasabsnDj96Yh9
"""

import numpy as np
import matplotlib.pyplot as plt

# Definir parámetros
D1 = D2 = .50  # cm
lambda_ = 670e-9  # Longitud de onda en metros
A = 0.4e-3  # Apertura en metros
a = 0.1e-3  # Ancho de rendija en metros
d = 0.1e-2  # Separación entre rendijas en metros
N = 100000  # Número de muestras aleatorias

# Generar muestras aleatorias dentro de la apertura
x = np.random.uniform(-A/2, A/2, N)
y = np.random.choice([-d/2, d/2], N) + np.random.uniform(-a/2, a/2, N)

# Evaluar la integral de Fresnel en distintos valores de z
z_values = np.linspace(-4e-3, 4e-3, 10000)  # Rango de z en metros
I = []

for z in z_values:
    phase = (np.pi / lambda_) * (x**2 / D1) + (np.pi / lambda_) * ((x - y)**2 / D2) + (np.pi / lambda_) * ((z - y)**2 / D1)
    integral = np.mean(np.exp(1j * phase))
    I.append(np.abs(integral)**2)

# Normalizar la intensidad
I /= np.max(I)

# Cálculo de la intensidad clásica
theta = np.arctan(z_values / D2)
I_classic = (np.cos((np.pi * d / lambda_) * np.sin(theta))**2) * (np.sinc(a * np.sin(theta) / lambda_)**2)
I_classic /= np.max(I_classic)  # Normalizar


# Graficar los resultados
plt.figure(figsize=(8, 5))
plt.plot(z_values, I, label='Difracción de Fresnel (Monte Carlo)')
plt.plot(z_values, I_classic, label='Modelo clásico', linestyle='dashed', color='r')
plt.xlabel('z (m)')
plt.ylabel('Intensidad Normalizada')
plt.title('Comparación entre Fresnel y Modelo Clásico')
plt.legend()
plt.grid()
plt.savefig("2.pdf")
plt.show()

