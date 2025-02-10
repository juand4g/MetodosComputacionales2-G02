import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from statsmodels.tsa.seasonal import seasonal_decompose

# Cargar los datos
archivo = open('C:/Users/Usuario/Desktop/manchas solares.txt', 'r')
year, month, day, manchas_sol = [], [], [], []

# Procesar el archivo
for linea in archivo:
    if linea.strip():  # Ignorar líneas vacías
        partes = linea.split()
        if len(partes) == 4 and partes[0] != 'Year' and int(partes[0]) < 2012:
            year.append(int(partes[0]))
            month.append(int(partes[1]))
            day.append(int(partes[2]))
            manchas_sol.append(int(partes[3]))
archivo.close()

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
plt.savefig('2.b.pdf')
plt.show()