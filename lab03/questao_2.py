import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para converter uma imagem colorida para tons de cinza usando a fórmula fornecida
def rgb_to_grayscale(img):
    R = img[:, :, 2]
    G = img[:, :, 1]
    B = img[:, :, 0]
    
    # Fórmula para converter para tons de cinza
    gray = 0.3 * R + 0.59 * G + 0.11 * B
    gray = np.uint8(gray)  # Convertendo para uint8 para exibir corretamente
    
    return gray

# Aplicar filtros convolucionais
def apply_filters(gray_img):
    # Filtro de média
    avg_blur = cv2.blur(gray_img, (5, 5))

    # Filtro Gaussiano
    gaussian_blur = cv2.GaussianBlur(gray_img, (5, 5), 1)

    # Filtro de Mediana
    median_blur = cv2.medianBlur(gray_img, 5)

    return avg_blur, gaussian_blur, median_blur

# Carregar a imagem colorida (substitua 'image_path' pelo caminho correto da sua imagem)
image_path = '/home/natalia/Música/praticas_visão/imagens/sub_imagens/brasil.jpg'
color_img = cv2.imread(image_path)

# Converter a imagem para tons de cinza usando a fórmula
gray_img = rgb_to_grayscale(color_img)

# Aplicar filtros convolucionais
avg_blur, gaussian_blur, median_blur = apply_filters(gray_img)

# Exibir as imagens
plt.subplot(221), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title('Imagem Colorida')
plt.subplot(222), plt.imshow(gray_img, cmap='gray'), plt.title('Imagem em Tons de Cinza')
plt.subplot(223), plt.imshow(avg_blur, cmap='gray'), plt.title('Filtro de Média')
plt.subplot(224), plt.imshow(gaussian_blur, cmap='gray'), plt.title('Filtro Gaussiano')

plt.show()
