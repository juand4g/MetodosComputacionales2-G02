import numpy as np
import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

file_name = os.path.join(script_dir,"catto.png")
ruta_corregida = file_name.replace("\\", "/")

img = cv2.imread(ruta_corregida)

print(file_name)
print(img)