#Imports =============================================================================
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import find_peaks, peak_widths
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
def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float],
  frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
  ts = np.arange(0.,t_max,dt)
  ys = np.zeros_like(ts,dtype=float)
  for A,f in zip(amplitudes,frecuencias):
    ys += A*np.sin(2*np.pi*f*ts)
    ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
  return ts,ys

t1,y1 = datos_prueba(28,0.05,[7,10,6],[0.5,0.6,0.4],ruido=7)
t2,y2 = datos_prueba(28,0.05,[7,10,6],[0.5,0.6,0.4])
y2=y2
plt.scatter(t1,y1,color="r")
plt.plot(t2,y2)

def Fourier(t:NDArray[float], y:NDArray[float],
 f:NDArray[float]) -> NDArray[complex]:
    N = len(y)
    result = np.zeros(len(f), dtype=complex)
    for k in range(N):
        result += y[k]*np.exp(-2j * np.pi * t[k] * f)
    return result

# Definimos las frecuencias a analizar
frecuencias_analisis = np.array(np.linspace(0.1,10,1000))
# Transformada de Fourier de las señales
ft_sin_ruido = Fourier(t2, y2, frecuencias_analisis)
ft_con_ruido = Fourier(t1, y1, frecuencias_analisis)

# Gráfico de las señales
plt.figure(figsize=(10, 6))
plt.plot(frecuencias_analisis, np.abs(ft_sin_ruido)/len(y1), label='Señal sin ruido')
plt.plot(frecuencias_analisis, np.abs(ft_con_ruido)/len(y1), label='Señal con ruido', alpha=0.7)
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Transformada de Fourier de las señales')
plt.savefig(os.path.join(script_dir,"1.a.pdf"))

print("1.a) Mantiene claro los picos, sin afectar mucho la señal")

# 1.b.

t3,y3 = datos_prueba(10,0.25,[21],[0.8])
frecuencias_analisis = np.array(np.linspace(0.1,2.5,1000))
ft_prueba = Fourier(t3, y3, frecuencias_analisis)
plt.figure(figsize=(10, 6))
plt.plot(frecuencias_analisis, np.abs(ft_prueba)/len(y3), label='Señal de prueba')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Transformada de Fourier de las señales')

# Encontramos los picos en la transformada sin ruido
peaks, properties = find_peaks(np.abs(ft_prueba)/len(y3), height=4 )

# Calculamos el FWHM
width = peak_widths(np.abs(ft_prueba)/len(y3),peaks,rel_height=0.5)

FWHM = np.diff(frecuencias_analisis)[0]*width[0]

# Mostramos el FWHM
print(f'FWHM del pico: {FWHM}')

# Graficamos el FWHM
plt.figure(figsize=(10, 6))
plt.plot(frecuencias_analisis, np.abs(ft_prueba)/len(y3), label='Transformada sin ruido')
plt.vlines(frecuencias_analisis[peaks], 0, np.abs(ft_prueba)[peaks]/len(y3), color='r', label='Picos')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.legend()
plt.title('Picos y FWHM de la Transformada de Fourier')

fwhm_values = []
t_max_values = np.logspace(1, 2, 100)  # valores de t_max entre 10 y 300 segundos
frecuencias_analisis = np.array(np.linspace(0.1,2.5,5000))

for t_max in t_max_values:
    t,y = datos_prueba(t_max,0.25,[21],[0.8])
    ft_prueba = Fourier(t, y, frecuencias_analisis)
    
    peaks, _ = find_peaks(np.abs(ft_prueba)/len(y), height=4)
    results_half = peak_widths(np.abs(ft_prueba)/len(y), peaks, rel_height=0.5)
    
    fwhm_values.append(results_half[0][0])

# Graficamos FWHM vs t_max
plt.figure(figsize=(10, 6))
plt.loglog(t_max_values, fwhm_values, ".-",label='FWHM vs t_max')
plt.xlabel('t_max (s)')
plt.ylabel('FWHM (Hz)')
plt.title('FWHM en función del intervalo de tiempo (log-log)')
plt.grid(True)
plt.savefig(os.path.join(script_dir,"1.b.pdf"))


# 1.c.
data = pd.read_csv("https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/phot/I/OGLE-LMC-CEP-0001.dat", delimiter=" ")

t = data.iloc[:, 0].values  # Primera columna: tiempo
y = data.iloc[:, 1].values  # Segunda columna: intensidad
y -= np.mean(y) #Se resta la media para centrar la señal

# Determinar frecuencia de Nyquist
delta_t = np.min(np.diff(t))  # Paso de muestreo mínimo
f_nyquist = 1 / (2 * delta_t)

# Calcular la Transformada de Fourier
f = np.linspace(0, f_nyquist, 500000)  # Frecuencias hasta Nyquist
data_fourier = Fourier(t, y, f)
data_fourier = np.abs(data_fourier) / len(y)


# Encontrar la frecuencia de oscilación f_true
peaks, _ = find_peaks(data_fourier)
f_true = f[peaks[np.argmax(data_fourier[peaks])]]  # Pico más alto

# Calcular la fase phi
phi = np.mod(f_true * t, 1)

# Graficar resultados
plt.figure(figsize=(15, 5))
plt.plot(f, data_fourier, label="Transformada de Fourier")
plt.axvline(f_true, color='r', linestyle='--', label=f"f_true = {f_true:.4f}")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.legend()
plt.title("Espectro de Fourier")

# Gráfico de dispersión de y vs phi
plt.figure(figsize=(10, 5))
plt.scatter(phi, y, alpha=0.5, s=10)
plt.xlabel("Fase φ")
plt.ylabel("Intensidad y")
plt.title("Gráfico de dispersión de y vs φ")

# Imprimir valores clave
print(f"1.c) f Nyquist: {f_nyquist:.4f}")
print(f"1.c) f true: {f_true:.4f}")

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

# Definir un filtro pasa-bajo (Butterworth)
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist  # Frecuencia de corte normalizada
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Coeficientes del filtro
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)  # Aplicar el filtro
    return y

# Filtrar los datos para eliminar el ruido (ajusta el corte según sea necesario)
# cutoff: frecuencia de corte, fs: frecuencia de muestreo (1 día por defecto)
cutoff = 0.01  # Frecuencia de corte (ajústala según el análisis, por ejemplo, 0.01 para eliminar frecuencias altas)
fs = 1  # Frecuencia de muestreo (1 día)
manchas_suavizadas = butter_lowpass_filter(df['manchas'].values, cutoff, fs)

# Visualizar los datos originales y los suavizados
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['manchas'], label='Datos originales')
plt.plot(df.index, manchas_suavizadas, label='Datos suavizados', linestyle='--', color='red')
plt.xlabel('Fecha')
plt.ylabel('Manchas solares')
plt.title('Datos de manchas solares con y sin suavizado')
plt.legend()

# Realizar la transformada de Fourier sobre los datos suavizados
frecuencias = np.fft.rfftfreq(len(manchas_suavizadas), d=1)  # 'd=1' significa separación por 1 día
transformada = np.fft.rfft(manchas_suavizadas)

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
#Cargar los datos
archivo_manchas = open(os.path.join(script_dir, "list_aavso-arssn_daily.txt"), 'r')
year, month, day, manchas_sol = [], [], [], []

# Procesar el archivo
for linea in archivo_manchas:
    if linea.strip():  # Ignorar líneas vacías
        partes = linea.split()
        if len(partes) == 4 and partes[0] != 'Year' and int(partes[0]) < 2012:
            year.append(int(partes[0]))
            month.append(int(partes[1]))
            day.append(int(partes[2]))
            manchas_sol.append(int(partes[3]))
archivo_manchas.close()

# Crear DataFrame con fechas
fechas = pd.to_datetime([f'{a}-{m}-{d}' for a, m, d in zip(year, month, day)])
df = pd.DataFrame({'fecha': fechas, 'manchas': manchas_sol})
df.set_index('fecha', inplace=True)

# Definir un filtro pasa-bajo (Butterworth)
def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    y = filtfilt(b, a, data)
    return y

# Filtrar los datos
cutoff, fs = 0.01, 1
manchas_suavizadas = butter_lowpass_filter(df['manchas'].values, cutoff, fs)

# Transformada de Fourier
frecuencias = np.fft.rfftfreq(len(manchas_suavizadas), d=1)
transformada = np.fft.rfft(manchas_suavizadas)

# Tomar solo los primeros 10 armónicos
n_armonicos = 10
transformada_filtrada = np.zeros_like(transformada)
transformada_filtrada[:n_armonicos] = transformada[:n_armonicos]

# Reconstrucción de la señal con los primeros 10 armónicos
prediccion = np.fft.irfft(transformada_filtrada)

# Extender la predicción hasta el 10 de febrero de 2025
fecha_inicio = df.index[0]
fecha_fin = pd.to_datetime("2025-02-10")
dias_prediccion = (fecha_fin - fecha_inicio).days

t = np.arange(len(manchas_suavizadas) + dias_prediccion)
prediccion_extendida = np.fft.irfft(transformada_filtrada, n=len(t))

# Obtener la predicción para el 10 de febrero de 2025
n_manchas_hoy = prediccion_extendida[-1]
print(f'2.b.b) {n_manchas_hoy = }')

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['manchas'], label='Datos originales', alpha=0.6)
plt.plot(pd.date_range(start=df.index[0], periods=len(prediccion_extendida)), prediccion_extendida, label='Predicción FFT (10 armónicos)', linestyle='--', color='red')
plt.xlabel('Fecha')
plt.ylabel('Número de manchas solares')
plt.title('Predicción de manchas solares usando FFT')
plt.legend()
plt.savefig(os.path.join(script_dir,'2.b.pdf'))

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
    img = cv2.imread(nombre, cv2.IMREAD_GRAYSCALE)  # Cargar en escala de grises
    return img.astype(np.float32) / 255.0  # Normalizar valores entre 0 y 1

def filtrar_ruido_periodico(img, umbral_relativo=0.1):
    filas, columnas = img.shape

    # Transformada de Fourier
    F = np.fft.fft2(img)
    F_shifted = np.fft.fftshift(F)

    # Magnitud del espectro
    magnitud = np.abs(F_shifted)

    # Identificar picos de alta intensidad en el espectro
    umbral = umbral_relativo * np.max(magnitud)
    mascara = magnitud < umbral  # Mantiene valores por debajo del umbral

    # Aplicar la máscara en el dominio de frecuencia
    F_shifted_filtrado = F_shifted * mascara

    # Transformada inversa
    F_inv = np.fft.ifftshift(F_shifted_filtrado)
    img_filtrada = np.fft.ifft2(F_inv).real

    return img_filtrada

# Procesar ambas imágenes
imagenes = ["catto.png", "Noisy_Smithsonian_Castle.jpg"]
umbrales = [0.01,0.05]
for i in range(2):
    img_name = imagenes[i]
    umbral = umbrales[i]
    img_path = os.path.join(script_dir,img_name)
    img = cargar_imagen(img_path)
    img_filtrada = filtrar_ruido_periodico(img, umbral)

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

    plt.savefig(os.path.join(script_dir,"filtered_"+img_name))





