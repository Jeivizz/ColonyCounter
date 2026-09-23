import cv2
import os
import numpy as np

os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts/truetype/dejavu"

os.environ.pop("XDG_SESSION_TYPE", None)

RSZ_COEF = 0.2 # Resizing coefficient
BORDER_SHRINK = 0.91 # Remove the petri plate border
TARGET_DIAM = 1000

img = cv2.imread("./data/test2.jpeg")


def find_petri(img):
    h, w = img.shape[:2]

    resized = cv2.resize(img, None, fx=RSZ_COEF, fy=RSZ_COEF)
    h_rsz, w_rsz = resized.shape[:2]

    rect = (
        int(w_rsz * 0.15),
        int(h_rsz * 0.15),
        int(w_rsz * 0.70),
        int(h_rsz * 0.70)
    )

    mask = np.zeros(resized.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(resized, mask, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)

    return int(cx / RSZ_COEF), int(cy / RSZ_COEF), int(radius / RSZ_COEF)

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
    mask_resized = cv2.resize(mask_crop, (crop_resized.shape[1], crop_resized.shape[0]), interpolation=cv2.INTER_NEAREST)

    return crop_resized, mask_resized


def treat_img(img):
    output, mask = crop_petri(img)

    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        minDist=6,
        dp=1.1,
        param1=150,
        param2=15,
        minRadius=0,
        maxRadius=15
    )

    if circles is None:
        return None

    circles  = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:

        cv2.circle(output, (x, y), r, (255, 0, 0), 4)

    count = str(circles.size)
    cv2.putText(output, count, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1)

    return output

crop, mask = crop_petri(img)

processed = treat_img(img)

view_resized = cv2.resize(processed, None, fx=0.6, fy=0.6)
original_resized = cv2.resize(crop, None, fx=0.6, fy=0.6)
cv2.imshow("test", view_resized)
cv2.imshow("original", original_resized)
cv2.waitKey(0)



