import cv2

def crop_image(image_path, y1, y2, x1, x2, use_path=False):
    if use_path:
        img = cv2.imread(image_path)
    else:
        img = image_path
    
    img_cropped = img[y1:y2, x1:x2] # done via y_start:y_end, x_start:x_end
    
    return img_cropped

def upscale_image(cropped_img):
    # Check if the image is grayscale
    if len(cropped_img.shape) == 2:
        # Convert the grayscale image to BGR
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)

    # Perform bicubic interpolation upscaling
    upscaled_bicubic = cv2.resize(cropped_img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # https://learnopencv.com/super-resolution-in-opencv/
    # if i want to use an upscaling algorithm -> slower and not as accurate for some reason

    resized = cv2.resize(upscaled_bicubic,dsize=None,fx=4,fy=4)

    return cropped_img, resized


def trim_white_space(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Otsu's thresholding method to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for each contour
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    # Calculate the combined bounding box points
    top_x = min([x for (x, y, w, h) in rects])
    top_y = min([y for (x, y, w, h) in rects])
    bottom_x = max([x+w for (x, y, w, h) in rects])
    bottom_y = max([y+h for (x, y, w, h) in rects])

    # Crop the image using the combined bounding box coordinates
    cropped = image[top_y:bottom_y, top_x:bottom_x]

    return cropped

def trim_black_space(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Otsu's thresholding method to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the bounding box for each contour
    rects = [cv2.boundingRect(cnt) for cnt in contours]

    # Calculate the combined bounding box points
    top_x = min([x for (x, y, w, h) in rects])
    top_y = min([y for (x, y, w, h) in rects])
    bottom_x = max([x+w for (x, y, w, h) in rects])
    bottom_y = max([y+h for (x, y, w, h) in rects])

    # Crop the image using the combined bounding box coordinates
    cropped = image[top_y:bottom_y, top_x:bottom_x]

    return cropped


def crop_image_cv2(imgz, left_percent, right_percent):
    img = imgz
    height, width = img.shape[:2]
    # crop image by a certain percent in order to remove other errors with image detection
    left = int(width * left_percent)
    right = int(width * (1 - right_percent))
    img_cropped = img[:, left:right]
    return img_cropped