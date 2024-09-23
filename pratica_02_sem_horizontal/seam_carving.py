import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, color

def calculate_energy(image):
    # Convertendo a imagem para escala de cinza
    gray_image = color.rgb2gray(image)
    # Calculando o gradiente da imagem
    energy = np.abs(filters.sobel_h(gray_image)) + np.abs(filters.sobel_v(gray_image))
    return energy

def find_seam(energy):
    # A energia acumulada ao longo do caminho mínimo
    r, c = energy.shape
    M = energy.copy()
    backtrack = np.zeros_like(M, dtype=int)

    # Preenchendo a matriz de energia acumulada
    for i in range(1, r):
        for j in range(c):
            # Bordas são tratadas separadamente
            if j == 0:
                idx = np.argmin(M[i-1, j:j+2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i-1, j-1:j+2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i-1, idx + j - 1]
            M[i, j] += min_energy

    return M, backtrack

def remove_seam_vertical(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r, c - 1, 3), dtype=image.dtype)
    j = np.argmin(backtrack[-1])
    for i in reversed(range(r)):
        output[i, :, 0] = np.delete(image[i, :, 0], [j])
        output[i, :, 1] = np.delete(image[i, :, 1], [j])
        output[i, :, 2] = np.delete(image[i, :, 2], [j])
        j = backtrack[i, j]
    return output

def remove_seam_horizontal(image, backtrack):
    r, c, _ = image.shape
    output = np.zeros((r - 1, c, 3), dtype=image.dtype)
    j = np.argmin(backtrack[:, -1])
    for i in reversed(range(c)):
        output[:, i, 0] = np.delete(image[:, i, 0], [j])
        output[:, i, 1] = np.delete(image[:, i, 1], [j])
        output[:, i, 2] = np.delete(image[:, i, 2], [j])
        j = backtrack[j, i]
    return output

def seam_carving(image, num_seams_vertical=0, num_seams_horizontal=0):
    # Remover costuras verticais
    for _ in range(num_seams_vertical):
        energy = calculate_energy(image)
        M, backtrack = find_seam(energy)
        image = remove_seam_vertical(image, backtrack)

    # Remover costuras horizontais (transposta da imagem)
    image = np.transpose(image, (1, 0, 2))  # Transpor a imagem para remover costuras horizontais como verticais
    for _ in range(num_seams_horizontal):
        energy = calculate_energy(image)
        M, backtrack = find_seam(energy)
        image = remove_seam_vertical(image, backtrack)
    image = np.transpose(image, (1, 0, 2))  # Transpor de volta à forma original

    return image

# Carregar a imagem
img = io.imread('/home/natalia/Música/praticas_visão/imagens/sub_imagens/balls.jpg')

# Aplicar o seam carving (remover costuras verticais e horizontais)
new_image = seam_carving(img, num_seams_vertical=40, num_seams_horizontal=30)

# Mostrar a imagem original e a modificada
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(img)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(new_image)
ax[1].set_title('Seam Carved Image (Vertical + Horizontal)')
ax[1].axis('off')

plt.tight_layout()
plt.show()
