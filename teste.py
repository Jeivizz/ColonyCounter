import cv2

img = cv2.imread('./data/test2.jpeg')

RSZ_COEF = 0.2

# 1. Converte para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv[:, :, 1]  # Canal de Saturação (cor pura)
v_channel = hsv[:, :, 2]  # Canal de Brilho

# 2. Em vez de Otsu puro na escala de cinza, combinamos o brilho e a saturação
# A placa é mais brilhante E mais saturada que o tecido cinza/suporte preto
score_map = cv2.addWeighted(s_channel, 0.6, v_channel, 0.4, 0)

# GaussianBlur para suavizar a textura do tecido de fundo
blurred = cv2.GaussianBlur(score_map, (15, 15), 0)

# Threshold adaptativo ou fixo no score_map
_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

# 3. Limpeza morfológica para preencher o interior da placa
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow('test', cv2.resize(thresh, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST))

cv2.waitKey(0)