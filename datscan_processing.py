import cv2
import numpy as np

# Convert image to grayscale
def preprocess_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Simple threshold-based segmentation
def segment_brain_regions(image_gray):
    blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image_gray.shape
    putamen_mask = np.zeros_like(image_gray)
    caudate_mask = np.zeros_like(image_gray)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 1000:
            x, y, _, _ = cv2.boundingRect(cnt)
            if x < w // 2:
                cv2.drawContours(putamen_mask, [cnt], -1, 255, -1)
            else:
                cv2.drawContours(caudate_mask, [cnt], -1, 255, -1)
    
    return putamen_mask, caudate_mask

def compute_uptake(image_gray, mask):
    masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    uptake = cv2.mean(image_gray, mask=mask)[0]
    return uptake
