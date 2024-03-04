import pytesseract as tess
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import numpy as np
import matplotlib.pyplot as plt

# use the location of pytesseract
tess.pytesseract.tesseract_cmd = r'C:\\Users\\harry\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

def readImage(source, whitelistedText):
    img = source
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Adaptive Thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Contour Detection
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_contour_area = 100  # Adjust this value based on your images
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    contour_image = np.zeros_like(binary)
    cv2.drawContours(contour_image, filtered_contours, -1, 255, thickness=cv2.FILLED)

    # Combine the original image with the contour image
    img = cv2.bitwise_and(img, img, mask=contour_image)

    # Dilation and Erosion
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)
    binary = cv2.erode(binary, kernel, iterations=1)

    # Adaptive Contrast and Sharpness Enhancement
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(3)

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(0.4)
    img = img.filter(ImageFilter.EDGE_ENHANCE)  # Enhancing

    # image back to grayscale
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    _, img_bw = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 170, 255, cv2.THRESH_BINARY)

    # use a certain mode to enhance text reading
    custom_config = r'--oem 3 --psm 11 outputbase digits -c tessedit_char_whitelist=' + whitelistedText
    text = tess.image_to_string(Image.fromarray(img_bw), config=custom_config)

    return text