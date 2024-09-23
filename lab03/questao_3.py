import cv2
import numpy as np
from matplotlib import pyplot as plt

# Função para aplicar o efeito sépia manualmente
def apply_sepia(img):
    # Copiar a imagem original para evitar modificar diretamente
    sepia_img = np.copy(img)

    # Aplicar a transformação sépia em cada canal
    for y in range(sepia_img.shape[0]):
        for x in range(sepia_img.shape[1]):
            r, g, b = sepia_img[y, x]
            tr = 0.393 * r + 0.769 * g + 0.189 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.272 * r + 0.534 * g + 0.131 * b

            # Atribuir os novos valores, limitando entre 0 e 255
            sepia_img[y, x] = [min(tb, 255), min(tg, 255), min(tr, 255)]

    return sepia_img

# Aplicar filtros convolucionais
def apply_filters(img):
    # Filtro de média
    avg_blur = cv2.blur(img, (5, 5))

    # Filtro Gaussiano
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 1)

    # Filtro de Mediana
    median_blur = cv2.medianBlur(img, 5)

    return avg_blur, gaussian_blur, median_blur

# Carregar a imagem colorida (substitua 'image_path' pelo caminho correto da sua imagem)
image_path = '/home/natalia/Música/praticas_visão/imagens/sub_imagens/brasil.jpg'
color_img = cv2.imread(image_path)

# Aplicar o efeito sépia manualmente
sepia_img = apply_sepia(color_img)

# Aplicar filtros convolucionais
avg_blur, gaussian_blur, median_blur = apply_filters(sepia_img)

# Exibir as imagens
plt.subplot(221), plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)), plt.title('Imagem Colorida Original')
plt.subplot(222), plt.imshow(cv2.cvtColor(sepia_img, cv2.COLOR_BGR2RGB)), plt.title('Imagem com Efeito Sépia')
plt.subplot(223), plt.imshow(cv2.cvtColor(avg_blur, cv2.COLOR_BGR2RGB)), plt.title('Filtro de Média')
plt.subplot(224), plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)), plt.title('Filtro Gaussiano')

plt.show()
