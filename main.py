import cv2
import os
import numpy as np

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

os.environ.pop("XDG_SESSION_TYPE", None)

RSZ_COEF = 0.2 # Resizing coefficient
BORDER_SHRINK = 0.92 # Remove the petri plate border
TARGET_DIAM = 1000

img = cv2.imread("./data/test.jpg")

def find_petri(img):
    resized = cv2.resize(img, None, fx=RSZ_COEF, fy=RSZ_COEF)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=100,
        param1=50,
        param2=30,
        minRadius=int(min(resized.shape[:2]) * 0.3),
        maxRadius=int(min(resized.shape[:2]) * 0.5)
    )

    output = resized.copy()
    h_img, w_img = resized.shape[:2]

    if circles is None:
        return None
    cx, cy, r = circles[0][0] / RSZ_COEF

    return int(cx), int(cy), int(r)

def crop_petri(img):

    circle = find_petri(img)
    if circle is None:
        return None

    cx, cy, r = circle
    r_noBorder = int(r * BORDER_SHRINK)
    h, w = img.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), r_noBorder, 255, -1)

    masked_img = cv2.bitwise_and(img, img, mask=mask)

    x0, y0 = max(0, cx - r_noBorder), max(0, cy - r_noBorder)
    x1, y1 = min(w, cx + r_noBorder), min(h, cy + r_noBorder)

    crop = masked_img[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]

    s = TARGET_DIAM / (2 * r_noBorder)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    crop_resized = cv2.resize(crop, None, fx=s, fy=s, interpolation=interp)
    mask_resized = cv2.resize(mask_crop, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    return crop_resized, mask_resized


crop, mask = crop_petri(img)

view_resized = cv2.resize(crop, None, fx=0.6, fy=0.6)
cv2.imshow("test", view_resized)
cv2.waitKey(0)



