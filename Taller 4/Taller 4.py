import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.animation as animation
import pandas as pd
import re
import random
import unidecode
from langdetect import detect
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
# Punto 4 ------------------------------------------------------------------------------------------------------------------------------------------------------
# Cadena de ADN de ejemplo
s = "GTCTTAAAGGGCGGGGTAAGGCCTTGTTCAACACTTGTCCCGTA"

# Definir los símbolos de ADN
atoms = list("ACGT")
chars = list("ACGT")

# Crear la tabla de frecuencias
F = pd.DataFrame(np.zeros((4, 4), dtype=int), index=atoms, columns=chars)

# Llenar la tabla de frecuencias
for i in range(len(s) - 1):
    F.loc[s[i], s[i + 1]] += 1

# Normalizar para obtener probabilidades condicionales
P = F / F.sum(axis=1).values[:, None]

# Función para predecir la siguiente letra
def generar_siguiente_letra(secuencia, P):
    ultima_letra = secuencia[-1]
    return np.random.choice(atoms, p=P.loc[ultima_letra])

# Ejemplo de generación
ejemplo = "ACG"
nueva_letra = generar_siguiente_letra(ejemplo, P)
print("Nueva letra generada:", nueva_letra)

# Abrir archivo y leer contenido
archivo = open('C:/Users/Usuario/Desktop/Libro.txt', 'r', encoding='utf-8')
lineas = archivo.readlines()
contenido = "".join(lineas)  # Convertir la lista a un solo string
contenido = contenido.replace("\r\n", "\n").replace("\n\n", "#").replace("\n", "").replace("#", "\n\n")
archivo.close()

# Eliminar caracteres especiales (tildes, comillas, guiones bajos)
contenido = unidecode.unidecode(contenido)  # Remueve tildes
contenido = re.sub(r"[^a-zA-Z0-9\s]", "", contenido)  # Deja solo letras y números

# Reducir múltiples espacios a solo uno
contenido = re.sub(r"\s+", " ", contenido)

# Convertir a minúsculas
contenido = contenido.lower().strip()

# Quitar primera y última parte del libro (opcional, si es de Gutenberg)
inicio = contenido.find("*** start of")  # Buscar inicio real del texto
fin = contenido.find("*** end of")  # Buscar fin del texto
if inicio != -1 and fin != -1:
    contenido = contenido[inicio:fin]


def entrenar_modelo(texto, n=3):
    ngramas = []
    siguientes = []

    for i in range(len(texto) - n):
        ngramas.append(texto[i:i + n])  # Extraer n-grama
        siguientes.append(texto[i + n])  # Caracter que sigue

    # Crear DataFrame con los n-gramas y sus siguientes caracteres
    df = pd.DataFrame({"n-grama": ngramas, "siguiente": siguientes})

    # Crear tabla de frecuencias
    F = df.pivot_table(index="n-grama", columns="siguiente", aggfunc="size", fill_value=0)

    # Normalizar para obtener probabilidades
    P = F.div(F.sum(axis=1), axis=0)

    return P


# Entrenar el modelo con n=3
modelo = entrenar_modelo(contenido, n=3)


# --- Generación de texto con DataFrame ---
def generar_texto(modelo, m=1500, n=3):
    # Filtrar los n-gramas que empiezan con \n
    n_gramas_con_salto = [ng for ng in modelo.index if ng.startswith("\n")]

    # Si no hay n-gramas que empiecen con \n, elegir uno aleatorio
    if n_gramas_con_salto:
        inicio = random.choice(n_gramas_con_salto)
    else:
        inicio = random.choice(modelo.index)

    texto_generado = inicio

    for _ in range(m):
        ultimo_ng = texto_generado[-n:]  # Último n-grama generado

        if ultimo_ng in modelo.index:
            probabilidades = modelo.loc[ultimo_ng].dropna()
            siguiente = np.random.choice(probabilidades.index, p=probabilidades.values)
        else:
            siguiente = " "  # Si no hay predicción, usar espacio

        texto_generado += siguiente

    return texto_generado


def es_ingles(palabra):
    try:
        return detect(palabra) == 'en'
    except:
        return False  # Si hay un error al detectar el idioma (p.ej., palabra vacía o no reconocida)


# Función para calcular el porcentaje de palabras en inglés
def calcular_porcentaje_ingles(texto):
    # Limpiar el texto, quitar puntuación y convertir a minúsculas
    palabras = re.findall(r'\b\w+\b', texto.lower())  # Esto extrae solo palabras

    # Contar las palabras en inglés
    palabras_ingles = [palabra for palabra in palabras if es_ingles(palabra)]

    # Calcular el porcentaje
    porcentaje = (len(palabras_ingles) / len(palabras)) * 100 if len(palabras) > 0 else 0
    return porcentaje
n_values = [3,4, 5, 6,7, 8]

# Bucle para generar texto para cada valor de n y calcular el porcentaje de inglés
for n in n_values:
    # Entrenar el modelo con el valor actual de n
    modelo = entrenar_modelo(contenido, n=n)

    # Generar texto con el modelo entrenado
    texto_generado = generar_texto(modelo, m=1500, n=n)

    # Guardar el texto generado en un archivo
    nombre_archivo = f'C:/Users/Usuario/Desktop/texto_generado_n_{n}.txt'
    with open(nombre_archivo, 'w', encoding='utf-8') as archivo_salida:
        archivo_salida.write(texto_generado)
        print(f"El texto generado con n={n} se ha guardado en '{nombre_archivo}'")

    # Calcular el porcentaje de palabras en inglés
    porcentaje_ingles = calcular_porcentaje_ingles(texto_generado)
    print(f"Porcentaje de palabras en inglés para n={n}: {porcentaje_ingles:.2f}%")
#Finalmente, tomamos los porcentajes hallados, respectivos de cada n, y lo graficamos.
#Un histograma es lo mas adecuado para visualizar los porcentajes de cada n.
porcentajes=[35.97,40.57,36.54,36.64,35.92,38.69]
plt.figure(figsize=(8, 6))
plt.bar(n_values, porcentajes, color='b', alpha=0.7, edgecolor='black')

# Agregar título y etiquetas
plt.title("Distribución del porcentaje de palabras en inglés para distintos valores de n", fontsize=14)
plt.xlabel("Valores de n", fontsize=12)
plt.ylabel("Porcentaje de palabras en inglés (%)", fontsize=12)

# Mostrar los valores encima de cada barra
for i, txt in enumerate(porcentajes):
    plt.text(n_values[i], porcentajes[i] + 0.5, f"{txt:.2f}%", ha='center', fontsize=12)

# Mostrar el histograma
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.show()
