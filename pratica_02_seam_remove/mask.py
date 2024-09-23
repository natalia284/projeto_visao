import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregar a imagem original
image_path = "/home/natalia/Música/praticas_visão/pratica_03_seam_remove/Example/test.png"  # Substituir pelo caminho correto da nova imagem
image = cv2.imread(image_path)

# Redimensionar a imagem para garantir que seja processada nas dimensões corretas (opcional)
image_resized = cv2.resize(image, (400, 400))

# Converter a imagem para o espaço de cores HSV
hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

# Definir os limites inferior e superior para a cor vermelha no espaço HSV
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])

# Criar a máscara para isolar a cor vermelha
mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

# Outros tons de vermelho no espaço HSV
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Combinar ambas as máscaras
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# Combinar as duas máscaras para cobrir a faixa completa do vermelho
mask = mask1 + mask2

# Aplicar operações morfológicas para remover pequenos artefatos
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# Suavizar a máscara para eliminar ruídos
mask_cleaned = cv2.GaussianBlur(mask_cleaned, (5, 5), 0)

# Inverter a máscara para deixar o cubo vermelho preto e o fundo branco
inverted_mask = cv2.bitwise_not(mask_cleaned)

# Mostrar a imagem original e a máscara invertida
plt.subplot(1, 2, 1)
plt.title('Imagem Original')
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title('Máscara Invertida (Cubo Vermelho Preto)')
plt.imshow(inverted_mask, cmap='gray')

plt.show()

# Salvar a máscara invertida gerada
cv2.imwrite("/home/natalia/Música/praticas_visão/pratica_03_seam_remove/Example/mascara.png", inverted_mask)
