import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
# Obtener la ruta absoluta del directorio donde está el archivo .py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Combinar la ruta del directorio con el nombre del archivo
file_path = os.path.join(script_dir, 'Hysteresis_org.dat')
archivo= open(file_path, 'r')
list_t = []
list_B = []
list_H = []
for linea in archivo:
    if linea != '':
            partes = linea.split()
            if len(partes) == 3:
                    num_fltT = float(partes[0])
                    num_fltB = float(partes[1])
                    num_fltH = float(partes[2])
                    list_t.append(num_fltT)
                    list_B.append(num_fltB)
                    list_H.append(num_fltH)

archivo.close()
#print("Lista T:", list_t)
#print("Lista B:", list_B)
#print("Lista H:", list_H)
plt.figure(figsize=(10, 5))  # Puedes ajustar el tamaño de la figura si lo deseas
plt.subplot(1, 2, 1)  # Esto crea una figura con 2 gráficos en una fila (1 fila, 2 columnas)
plt.plot(list_t, list_B, color='b', marker='o', linestyle='-', label='t vs B')  # Graficamos t vs B
plt.title('t vs B')
plt.xlabel('t')  # Etiqueta del eje x
plt.ylabel('B')  # Etiqueta del eje y
plt.grid(True)  # Agregar una cuadrícula
plt.legend()
plt.show()
#Para hallar la frecuencia lo que hice fue plantear una curva senosoidal que se ajuste
#bien a los datos y, a partir de los parametros de la curva de ajuste, tomar la frecuencia
#de la curva de ajuste.
# Definir la función seno para el ajuste
def func_seno(t, A, f, phi, C):
    return A * np.sin(2 * np.pi * f * t + phi) + C

# Ajuste de la curva
popt, _ = curve_fit(func_seno, list_t, list_B, p0=[2.8, 1/2, 0, -0.10])
# p0 son los valores iniciales de amplitud, frecuencia y desplazamiento vertical y horizontal
# que estimo yo a partir de la grafica del primer punto.
# Extraer los parámetros ajustados
A_fit, f_fit, phi_fit, C_fit = popt


# Mostrar los resultados

print(f"Frecuencia (f): {f_fit}")

# Crear la segunda gráfica (t vs H)
plt.subplot(1, 2, 2)  # Esto coloca el gráfico en la segunda posición
plt.plot(list_t, list_H, color='r', marker='x', linestyle='-', label='t vs H')  # Graficamos t vs H
plt.title('t vs H')
plt.xlabel('t')  # Etiqueta del eje x
plt.ylabel('H')  # Etiqueta del eje y
plt.grid(True)  # Agregar una cuadrícula
plt.legend()
#plt.savefig("histerico.pdf", format="pdf")
plt.show()