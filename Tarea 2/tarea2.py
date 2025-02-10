#Imports =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import savgol_filter
import os
import cv2

# Obtener la ruta absoluta del directorio donde está el archivo .py
script_dir = os.path.dirname(os.path.abspath(__file__))

# Punto 1: Transformada general ======================================================
# 1.a.

# 1.b.

# 1.c.

# Punto 2: Transformada rápida ==========================================================
# 2.a.

# Cargar datos
H_field = open(os.path.join(script_dir, "H_field.csv"), 'r')
data = pd.read_csv(H_field)

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
plt.savefig(os.path.join(script_dir,"2.a.pdf"))

# 2.b.a. ==========================================================================

# Cargar los datos
archivo_manchas = open(os.path.join(script_dir, "list_aavso-arssn_daily.txt"), 'r')
year = []
month = []
day = []
manchas_sol = []

# Procesar el archivo
for linea in archivo_manchas:
    if linea != '':  # Ignorar líneas vacías
        partes = linea.split()  # Separar cada parte de la línea
        if len(partes) == 4:  # Asegurarse de que la línea tenga 4 partes
            if partes[0] != 'Year':  # Ignorar encabezado
                if int(partes[0]) < 2012:  # Filtrar datos hasta 2012
                    num_fltY = int(partes[0])
                    num_fltM = int(partes[1])
                    num_fltD = int(partes[2])
                    num_fltS = int(partes[3])
                    year.append(num_fltY)
                    month.append(num_fltM)
                    day.append(num_fltD)
                    manchas_sol.append(num_fltS)
archivo_manchas.close()  # Cerrar el archivo después de leer

fechas = pd.to_datetime([f'{a}-{m}-{d}' for a, m, d in zip(year, month, day)])
datos_manchas = np.array(manchas_sol)
plt.plot(fechas, datos_manchas)
plt.xlabel('Fecha')
plt.ylabel('Número de manchas solares')
plt.title('Manchas solares a lo largo del tiempo')
df = pd.DataFrame({'fecha': fechas, 'manchas': manchas_sol})
df.set_index('fecha', inplace=True)

# Aplicar un promedio móvil (ejemplo: ventana de 30 días)
ventana = 30  # Tamaño de la ventana del promedio móvil
manchas_suavizadas = df['manchas'].rolling(window=ventana, center=True).mean()
manchas_suavizadas_sg = savgol_filter(df['manchas'], window_length=51, polyorder=3)

# Visualizar los datos originales y los suavizados
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['manchas'], label='Datos originales')
plt.plot(df.index, manchas_suavizadas_sg, label='Savitzky-Golay', linestyle='--', color='red')
plt.xlabel('Fecha')
plt.ylabel('Manchas solares')
plt.title('Datos de manchas solares con y sin suavizado (Savitzky-Golay)')
plt.legend()

# Realizar la transformada de Fourier sobre los datos suavizados
frecuencias = np.fft.rfftfreq(len(manchas_suavizadas_sg), d=1)  # 'd=1' significa separación por 1 día
transformada = np.fft.rfft(manchas_suavizadas_sg)

# Calcular la densidad espectral
densidad_espectral = np.abs(transformada)**2

# Visualizar la densidad espectral en escala log-log
plt.figure(figsize=(8, 6))
plt.loglog(frecuencias, densidad_espectral, label='Densidad espectral')
plt.xlabel('Frecuencia [1/día]', fontsize=12)
plt.ylabel('Densidad espectral', fontsize=12)
plt.title('Transformada de Fourier de las manchas solares (Escala log-log)', fontsize=14)
plt.grid(True)
plt.legend()

# Encontrar la frecuencia dominante y evitamos el indice 0
frecuencia_dominante = frecuencias[np.argmax(np.abs(transformada[1:]))+1]

# Calcular el período (inverso de la frecuencia dominante en días)
if frecuencia_dominante != 0:
    periodo_solar_dias = 1 / frecuencia_dominante
    periodo_solar_anos = periodo_solar_dias / 365.25
    print(f'2.b.a) P_solar = {periodo_solar_anos:.2f} años')
else:
    print("2.b.a) No se detectó una frecuencia dominante significativa.")


# 2.b.b. ==========================================================================

# Punto 3: Filtros ===============================================================
# 3.a 
solar_file =  os.path.join(script_dir, "list_aavso-arssn_daily.txt")
solar_df = pd.read_csv(solar_file, delim_whitespace=True, skiprows=1, header=0)
solar_df = solar_df.dropna(subset=["SSN"])
solar_df["Date"] = pd.to_datetime(solar_df[["Year", "Month", "Day"]])


# Calcular el intervalo de tiempo medio entre muestras en días
dt = np.mean(np.diff(solar_df["Date"]).astype("timedelta64[D]")).astype(float)

def filtro(f, a):
    return np.exp(-(f*a)**2)

def aplicar_filtro(signal, dt, a):
    N = len(signal)
    freqs = np.fft.fftfreq(N, d=dt)

    S_f = np.fft.fft(signal)

    S_f_filtrada = S_f * filtro(freqs, a)

    senial_filtrada = np.fft.ifft(S_f_filtrada).real

    return senial_filtrada, np.abs(S_f_filtrada)

# Aplicar el filtro a la columna SSN
# Valores de α a probar
a_values = [1,8,16,32,64]  # Desde casi sin filtrar hasta casi borrar la señal

# Crear figura con subgráficos
fig, axes = plt.subplots(len(a_values), 2, figsize=(10, 8))

for i, a in enumerate(a_values):
    ssn_filtrado, S_f_filtrada = aplicar_filtro(solar_df["SSN"].values, dt, a)
    
    # Gráfica de la señal en el tiempo
    axes[i, 0].plot(solar_df["Date"], solar_df["SSN"], label="Original", alpha=0.7)
    axes[i, 0].plot(solar_df["Date"], ssn_filtrado, label=f"Filtrado (α={a})", linewidth=2)
    axes[i, 0].legend()
    
    # Gráfica de la transformada de Fourier
    freqs = np.fft.fftfreq(len(solar_df["SSN"]), d=dt)
    axes[i, 1].plot(freqs[:len(freqs)//2], np.abs(np.fft.fft(solar_df["SSN"]))[:len(freqs)//2], label="Original", alpha=0.7)
    axes[i, 1].plot(freqs[:len(freqs)//2], S_f_filtrada[:len(freqs)//2], label=f"Filtrado (α={a})", linewidth=2)
    axes[i, 1].legend()
    
    # Agregar texto con el valor de α
    axes[i, 0].text(0.05, 0.85, f"α = {a}", transform=axes[i, 0].transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Etiquetas de ejes
for ax in axes[:, 0]:
    ax.set_ylabel("SSN")
for ax in axes[:, 1]:
    ax.set_ylabel("FFT Magnitud")
    ax.set_xlabel("Frecuencia (1/días)")

axes[0, 0].set_title("Señal en el tiempo")
axes[0, 1].set_title("Transformada de Fourier")

plt.tight_layout()
plt.savefig(os.path.join(script_dir, "3.1.pdf"))  # Guardar la figura en PDF

#3.b ==========================================================================
# Función para cargar y convertir imagen a escala de grises
def cargar_imagen(nombre):
    img = cv2.imread(nombre)  # Cargar en escala de grises
    return img.astype(np.float32) / 255.0  # Normalizar valores entre 0 y 1

# Función para aplicar filtro en la Transformada de Fourier
def filtrar_ruido_periodico(img, umbral=50):
    # Aplicar FFT 2D
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)  # Centrar el espectro

    # Obtener la magnitud del espectro de Fourier
    magnitud = np.log1p(np.abs(F_shifted))

    # Crear una máscara para eliminar picos (ruido periódico)
    filas, columnas = img.shape
    cx, cy = columnas // 2, filas // 2  # Centro de la imagen
    mask = np.ones((filas, columnas), np.uint8)

    # Identificar picos brillantes fuera del centro y suprimirlos
    mask[magnitud > umbral] = 0  # Elimina valores altos (picos de ruido)

    # Aplicar la máscara a la transformada de Fourier
    F_shifted *= mask

    # Transformada Inversa
    F_inverse = np.fft.ifftshift(F_shifted)
    img_filtrada = np.fft.ifft2(F_inverse).real

    return img_filtrada

# Procesar ambas imágenes
imagenes = [os.path.join(script_dir,"catto.png"), os.path.join(script_dir,"Noisy_Smithsonian_Castle.jpg")]

for img_name in imagenes:
    print(f"Intentando cargar la imagen desde: {img_name}")
    img = cargar_imagen(img_name)
    img_filtrada = filtrar_ruido_periodico(img, umbral=50)

    # Mostrar resultados
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original - {img_name}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_filtrada, cmap="gray")
    plt.title(f"Filtrada - {img_name}")
    plt.axis("off")

    plt.show()





