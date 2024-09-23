import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor

# Verifique o caminho da imagem
image_path = '/home/natalia/Música/praticas_visão/imagens/pontos_ransac.png' 

# Carregar a imagem em escala de cinza e em RGB para exibição
img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread(image_path)  # Carregar em colorido para plotar a reta

# Verificar se a imagem foi carregada corretamente
if img_gray is None or img_color is None:
    print(f"Erro: Não foi possível carregar a imagem no caminho: {image_path}")
    exit()

# Aplicar um threshold para transformar a imagem em binária (pontos destacados)
_, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Encontrar os contornos dos pontos na imagem
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extrair as coordenadas dos contornos como pontos (x, y)
points = []
for cnt in contours:
    for c in cnt:
        points.append(c[0])

# Verificar se os pontos foram extraídos corretamente
if len(points) == 0:
    print("Erro: Nenhum ponto foi encontrado na imagem.")
    exit()

# Converter para um array NumPy
points = np.array(points)

# Separar os pontos em coordenadas X e Y
X = points[:, 0].reshape(-1, 1)  # Coordenadas X
Y = points[:, 1]  # Coordenadas Y

# Ajustar o modelo RANSAC para detectar a linha
ransac = RANSACRegressor()
ransac.fit(X, Y)

# Obter a linha estimada pelo RANSAC
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_Y_ransac = ransac.predict(line_X)

# Desenhar a linha ajustada na imagem original
pt1 = (int(line_X[0]), int(line_Y_ransac[0]))  # Ponto inicial da linha
pt2 = (int(line_X[-1]), int(line_Y_ransac[-1]))  # Ponto final da linha
cv2.line(img_color, pt1, pt2, (0, 0, 255), 2)  # Desenhar linha vermelha (RGB) na imagem

# Exibir a imagem original com a linha ajustada
cv2.imshow('Imagem com Reta Ajustada (RANSAC)', img_color)
cv2.waitKey(0)  # Esperar uma tecla para fechar
cv2.destroyAllWindows()

# Alternativamente, você pode salvar a imagem com a linha
# cv2.imwrite('/caminho/para/salvar/imagem_com_linha.png', img_color)
