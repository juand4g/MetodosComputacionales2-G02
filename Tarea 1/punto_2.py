import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import re
# Obtener la ruta absoluta del directorio donde está el archivo .py
script_dir = os.path.dirname(os.path.abspath(__file__))

#========================================================
#Lectura del archivo original
file_path = os.path.join(script_dir, 'hysteresis.dat')

archivo= open(file_path, 'r+')
lineas=archivo.readlines()
for i in range(0,len(lineas)):
    lista= list(lineas[i])
    for j in range(1,len(lineas[i])-1):
        if lista[j] =='-':
            lista[j]= ' -'
        elif lista[j] == '0' and lista[j+1] == '.' and lista[j-1] != ' -' and lista[j-1] != '1':
            lista[j]= ' 0'
    lineas[i] = ''.join(lista)  # Convertimos la lista de nuevo a una cadena

# Volvemos al principio del archivo para sobrescribirlo
archivo.seek(0)
archivo.writelines(lineas)  # Escribimos las líneas modificadas en el archivo
archivo.close()
#Volvemos a abrirlo ahora ya con las modificaciones para poder leer los datos
archivo= open(file_path, 'r')
list_t = []
list_B = []
list_H = []
for linea in archivo:
    if linea != '':
            linea= re.sub(r'\s+', ' ', linea).strip()
            partes = linea.split()
            if len(partes) == 3:
                    num_fltT = float(partes[0])
                    num_fltB = float(partes[1])
                    num_fltH = float(partes[2])
                    list_t.append(num_fltT)
                    list_B.append(num_fltB)
                    list_H.append(num_fltH)

archivo.close()
#===========================================================================

#print("Lista T:", list_t)
#print("Lista B:", list_B)
#print("Lista H:", list_H)
plt.figure(figsize=(10, 5))  # Puedes ajustar el tamaño de la figura si lo deseas
plt.subplot(1, 2, 1)  # Esto crea una figura con 2 gráficos en una fila (1 fila, 2 columnas)
plt.plot(list_t, list_B, color='b', marker='o', linestyle='-', label='t vs B')  # Graficamos t vs B
plt.title('Campo externo en función del tiempo.')
plt.xlabel('Tiempo (ms)')  # Etiqueta del eje x
plt.ylabel('Campo externo (mT)')  # Etiqueta del eje y
plt.grid(True)  # Agregar una cuadrícula
plt.legend()
#plt.show()
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

print(f"2.b) Frecuencia (f): {f_fit} Hz. Método: Ajuste de función sinusoidal.")

# Crear la segunda gráfica (t vs H)
plt.subplot(1, 2, 2)  # Esto coloca el gráfico en la segunda posición
plt.plot(list_t, list_H, color='r', marker='x', linestyle='-', label='t vs H')  # Graficamos t vs H
plt.title('Densidad de campo interno en función del tiempo.')
plt.xlabel('Tiempo (ms)')  # Etiqueta del eje x
plt.ylabel('Densidad de campo interno (A/m)')  # Etiqueta del eje y
plt.grid(True)  # Agregar una cuadrícula
plt.legend()
#plt.show()
plt.savefig(os.path.join(script_dir, 'histerico.pdf'), format="pdf")

#==============================================================================================
#PUNTO 2C 
fig, ax = plt.subplots()

# Crear la gráfica parametrizada por t usando scatter y un colormap
scatter = ax.scatter(list_B, list_H, c=list_t, cmap='viridis', edgecolor='k')

# Agregar etiquetas a los ejes
ax.set_xlabel('B (mT)')
ax.set_ylabel('H (A/m)')
ax.set_title('Gráfica de H vs B parametrizada por el tiempo.')

# Agregar una barra de color para mostrar los valores de t
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Tiempo (ms)')


fig.savefig(os.path.join(script_dir, 'energy.pdf'), format="pdf")


#CÁLCULO DEL ÁREA 

# Convertir a arrays de NumPy para cálculos eficientes
B = np.array(list_B)
H = np.array(list_H)

# Calcular el área usando la fórmula del polígono de Shoelace
area = 0.5 * np.abs(np.dot(B, np.roll(H, 1)) - np.dot(H, np.roll(B, 1)))

print(f"2.c) Pérdida de energía: {area:.4f} mJ.")


