import numpy as np
from keras.models import load_model
from CropImage import *

def predict(model, data):
    return model.predict(data)

def is_empty(image):
    avg_pixel_value = np.mean(image)
    return 250 <= avg_pixel_value <= 255

model_pathsZ = ['models/model_X_G_final.keras', 
               'models/model_X_A_final.keras', 
               'models/model_X_M_final.keras', 
               'models/model_X_K_final.keras', 
               'models/model_X_L_final.keras',]

modelsR = [load_model(path) for path in model_pathsZ]

def IFENeuralNetwork(source, gamlovalue):

    img =  cv2.imread("PDFScan/sample"+str(source)+".jpg")
    croppedimg = crop_image(img, 447, 447+128, 23, 23+267, use_path=False)
    croppedimg = trim_white_space(croppedimg)

    height, width = croppedimg.shape[:2]
    croppedimg = crop_image(croppedimg, round(height * 0.15), height-3, 8, width-3, use_path=False)

    croppedimg = cv2.GaussianBlur(croppedimg, (5, 5), cv2.BORDER_DEFAULT)

    croppedimg = upscale_image(croppedimg)[1]

    if is_empty(croppedimg):
        print("\033[91m Could not find IFE! Skipping IFE Scan... \033[0m")
        return False

    min_brightness_values = []

    for i in range(0, croppedimg.shape[1], croppedimg.shape[1] // 6):
        min_brightness = np.min(croppedimg[:, i:i + croppedimg.shape[1] // 6])
        min_brightness_values.append(min_brightness)

    X_G, X_A, X_M, X_K, X_L = [], [], [], [], []

    for i, min_brightness in enumerate(min_brightness_values):
        if min_brightness >= 20:
            if i == 1:
                X_G.append((round(min_brightness / 255, 12), float(gamlovalue)))
            elif i == 2:
                X_A.append((round(min_brightness / 255, 12), float(gamlovalue)))
            elif i == 3:
                X_M.append((round(min_brightness / 255, 12), float(gamlovalue)))
            elif i == 4:
                X_K.append((round(min_brightness / 255, 12), float(gamlovalue)))
            elif i == 5:
                X_L.append((round(min_brightness / 255, 12), float(gamlovalue)))

    data_X = [np.array(X_G), np.array(X_A), np.array(X_M), np.array(X_K), np.array(X_L)]

    predictions = [predict(model, data) for model, data in zip(modelsR, data_X)]

    binary_predictions = [[1 if pred >= 0.5 else 0 for pred in prediction] for prediction in predictions]

    current_diagnosis = []

    for i, binary_prediction in enumerate(binary_predictions):
        current_diagnosis.append((('g', 'a', 'm', 'kappa', 'lambda')[i], binary_prediction))

    diagnoseGloblins = []

    for diagnosis in current_diagnosis:
        if diagnosis[1][0] == 1:
            diagnoseGloblins.append(diagnosis[0])

    if len(diagnoseGloblins) != 0:
        if 'kappa' not in diagnoseGloblins and 'lambda' not in diagnoseGloblins:
            if X_K[0] < X_L[0] and X_K[0] != 0:
                diagnoseGloblins.append('kappa')
            else:
                diagnoseGloblins.append('lambda')

    if len(diagnoseGloblins) != 2:  # Ensure there are exactly two diagnoses
        diagnoseGloblins = []

    if "g" in diagnoseGloblins and "lambda" in diagnoseGloblins or "g" in diagnoseGloblins and "kappa" in diagnoseGloblins:
        if "g" in diagnoseGloblins:
            g_index = diagnoseGloblins.index("g")
            if -60 < current_diagnosis[g_index][1][0] - min_brightness_values[-1] < 150:
                diagnoseGloblins = []
    if "a" in diagnoseGloblins and "lambda" in diagnoseGloblins:
        if "a" in diagnoseGloblins:
            a_index = diagnoseGloblins.index("a")
            if -30 < current_diagnosis[a_index][1][0] - min_brightness_values[2] < 25:
                diagnoseGloblins = []
    if "a" in diagnoseGloblins and "lambda" in diagnoseGloblins:
        if "a" in diagnoseGloblins:
            a_index = diagnoseGloblins.index("a")
            if min_brightness_values[2] - current_diagnosis[-1][1][0] < -101:
                diagnoseGloblins = ['a', 'kappa']
    if "m" in diagnoseGloblins and "kappa" in diagnoseGloblins:
        if "m" in diagnoseGloblins:
            m_index = diagnoseGloblins.index("m")
            if -2 < current_diagnosis[m_index][1][0] - min_brightness_values[-2] < 5:
                diagnoseGloblins = []
    if "m" in diagnoseGloblins and "lambda" in diagnoseGloblins:
        if "m" in diagnoseGloblins:
            m_index = diagnoseGloblins.index("m")
            if 15 < current_diagnosis[m_index][1][0] - min_brightness_values[-1]:
                diagnoseGloblins = []

    return diagnoseGloblins
