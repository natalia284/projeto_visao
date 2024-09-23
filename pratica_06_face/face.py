import cv2
import numpy as np
from retinaface import RetinaFace
from ultralytics import YOLO
import os

# Função para calcular IoU entre dois bounding boxes
def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calcular as coordenadas da interseção
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)

    # Calcular área de interseção
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Áreas dos bounding boxes
    box1Area = w1 * h1
    box2Area = w2 * h2

    # Calcular IoU
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou

# Crie o diretório de saída, se não existir
output_directory = "/home/natalia/Música/praticas_visão/imagens/output/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Carregar Haar Cascade a partir do arquivo baixado
face_cascade = cv2.CascadeClassifier('/home/natalia/Música/praticas_visão/pratica_07_face/haarcascade_frontalface_default.xml')

# Carregar o modelo YOLOv8 pré-treinado para detecção de faces
model_yolo = YOLO("/home/natalia/Música/praticas_visão/pratica_07_face/yolov8n-face.pt")  # YOLOv8 Nano

# Função para detectar faces e desenhar bounding boxes
def detect_and_draw_faces(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # HAAR Cascade
    faces_haar = face_cascade.detectMultiScale(gray, 1.1, 4)

    # YOLOv8
    results = model_yolo(img)
    faces_yolo = results[0].boxes.xywh.cpu().numpy()  # YOLO retorna bounding boxes no formato (x_center, y_center, width, height)
    faces_yolo = [[int(x - w/2), int(y - h/2), int(w), int(h)] for (x, y, w, h) in faces_yolo]

    # RetinaFace
    faces_retina = RetinaFace.detect_faces(img_path)
    faces_retina = [[int(faces_retina[key]['facial_area'][0]), int(faces_retina[key]['facial_area'][1]),
                     int(faces_retina[key]['facial_area'][2] - faces_retina[key]['facial_area'][0]),
                     int(faces_retina[key]['facial_area'][3] - faces_retina[key]['facial_area'][1])]
                    for key in faces_retina]

    # Desenhar bounding boxes nas imagens
    for (x, y, w, h) in faces_haar:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Azul para Haar Cascade

    for (x, y, w, h) in faces_yolo:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Verde para YOLO

    for (x, y, w, h) in faces_retina:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Vermelho para RetinaFace

    # Salvar a imagem com bounding boxes desenhados
    output_path = os.path.join(output_directory, f"output_{os.path.basename(img_path)}")
    if cv2.imwrite(output_path, img):
        print(f"Imagem salva com sucesso em {output_path}")
    else:
        print(f"Erro ao salvar a imagem em {output_path}")

    # Calcular IoU entre os métodos
    iou_haar_yolo = 0
    iou_haar_retina = 0
    num_comparisons_yolo = 0
    num_comparisons_retina = 0

    # Comparar Haar com YOLO
    for box1 in faces_haar:
        for box2 in faces_yolo:
            iou_haar_yolo += calculate_iou(box1, box2)
            num_comparisons_yolo += 1

    # Comparar Haar com RetinaFace
    for box1 in faces_haar:
        for box2 in faces_retina:
            iou_haar_retina += calculate_iou(box1, box2)
            num_comparisons_retina += 1

    # Calcular a média de IoU
    avg_iou_haar_yolo = iou_haar_yolo / num_comparisons_yolo if num_comparisons_yolo > 0 else 0
    avg_iou_haar_retina = iou_haar_retina / num_comparisons_retina if num_comparisons_retina > 0 else 0

    print(f"Média de IoU Haar vs YOLO para {img_path}: {avg_iou_haar_yolo}")
    print(f"Média de IoU Haar vs RetinaFace para {img_path}: {avg_iou_haar_retina}")

# Lista de imagens para teste
image_paths = [
    '/home/natalia/Música/praticas_visão/imagens/fotinha.jpeg',
    '/home/natalia/Música/praticas_visão/imagens/fotinha2.jpeg',
    '/home/natalia/Música/praticas_visão/imagens/fotinha3.jpeg'
]

# Processar cada imagem e comparar os métodos
for img_path in image_paths:
    print(f"Processando {img_path}...")
    detect_and_draw_faces(img_path)
