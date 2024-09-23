import sys
import cv2
import numpy as np

# Carregar uma imagem
image = cv2.imread('/home/natalia/Música/praticas_visão/imagens/sub_imagens/museum1.jpg', cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if image is None:
    print("Erro ao carregar a imagem!")
    sys.exit(1)

##############################
# SURF
##############################
# Verificar se o SURF está disponível
if hasattr(cv2, 'xfeatures2d'):
    # Criar o objeto SURF
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=1000)

    # Detectar key points e calcular descritores
    keypoints, descriptors = surf.detectAndCompute(image, None)

    # Desenhar key points na imagem
    img_surf = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Exibir imagem com keypoints do SURF
    cv2.imshow('SURF', img_surf)
else:
    print("SURF não está disponível na sua versão do OpenCV. Certifique-se de que foi compilado com OPENCV_ENABLE_NONFREE=ON.")

##############################
# ORB
##############################

# Criar o objeto ORB
orb = cv2.ORB_create()

# Detectar key points
keypoints = orb.detect(image, None)

# Calcular descritores ORB
keypoints, descriptors = orb.compute(image, keypoints)

# Desenhar keypoints
img_orb = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# Exibir imagem com keypoints do ORB
cv2.imshow('ORB', img_orb)

# Aguardar a tecla de fechamento
cv2.waitKey(0)
cv2.destroyAllWindows()
