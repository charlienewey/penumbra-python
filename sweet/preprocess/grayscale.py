import cv2

def to_grayscale(imgs):
    return [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
