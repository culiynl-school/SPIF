import pandas as pd
import numpy as np
from keras.models import load_model
from CropImage import *

model_pathsZ = ['models/model_X_G_final.keras', 
               'models/model_X_A_final.keras', 
               'models/model_X_M_final.keras', 
               'models/model_X_K_final.keras', 
               'models/model_X_L_final.keras',]

modelsR = []
for path in model_pathsZ:
    model = load_model(path)
    modelsR.append(model)

def scan_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    percentages = [0.44, 0.55, 0.66]
    
    indices = [int(p * gray.shape[1]) for p in percentages]
    
    brightness_averages = []
    
    # Iterate over the image at specific intervals
    for i in range(gray.shape[0]):
        brightness_sum = 0
        count = 0
        
        for j in indices:
            brightness = gray[i, j]
            
            if 0 <= brightness <= 255: # just in case
                brightness_sum += brightness
                count += 1
        
        brightness_averages.append(brightness_sum / count if count > 0 else 0)
    return brightness_averages
    
def is_empty(image):
    avg_pixel_value = np.mean(image)
    return 250 <= avg_pixel_value <= 255

def split_image(img, num_sections):
    imgheight, imgwidth, _ = img.shape
    section_width = imgwidth // num_sections
    for i in range(0, imgwidth, section_width):
        if i // section_width < num_sections:
            yield img[:, i:i+section_width]

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

    labels = ['sp', 'g', 'a', 'm', 'kappa', 'lambda']
    min_brightness_values = []

    df = pd.DataFrame()

    # Iterate over each section of the image
    for i, section in enumerate(split_image(croppedimg, 6)):
        # Calculate brightness averages
        brightness_averages = scan_brightness(section)
        
        temp_df = pd.DataFrame({
            'Brightness': brightness_averages,
            'x': [i for i in range(len(brightness_averages))],
            # 'Diagnosis': diagnosis,
            'Label': labels[i]
        })
        
        df = df._append(temp_df, ignore_index=True)

    df.to_csv("peaks/brightness_averages_" + str(source) + ".csv", index=False)

    # create seperate arrays for each of the different data values depending on its labels
    X_G = [] 
    X_A = []
    X_M = []
    X_K = []
    X_L = []

    x = source
    # print (f"peaks/brightness_averages_{x}.csv")

    data = pd.read_csv(f"peaks/brightness_averages_{x}.csv")

    brightness_values_saved = []

    for value in labels:
        for i in range(len(data)): 
            extracted_data = data[data['Label'] == value][['Brightness']].iloc[i:]
            min_brightness_values = extracted_data['Brightness'].min()
            max_brightness_values = 255

            brightness_values_saved.append(min_brightness_values)

            if min_brightness_values >= 20:
                if value == "g":
                    X_G.append((round(min_brightness_values/max_brightness_values,12), float(gamlovalue)))
                elif value == "a":
                    X_A.append((round(min_brightness_values/max_brightness_values,12), float(gamlovalue)))
                elif value == "m":
                    X_M.append((round(min_brightness_values/max_brightness_values,12), float(gamlovalue)))
                elif value == "kappa":
                    X_K.append((round(min_brightness_values/max_brightness_values,12), float(gamlovalue)))
                elif value == "lambda":
                    X_L.append((round(min_brightness_values/max_brightness_values,12), float(gamlovalue)))
                break  # Exit the loop once a suitable value is found

    data_X = [np.array(X_G), np.array(X_A), np.array(X_M), np.array(X_K), np.array(X_L)]

    predictions = [model.predict(data) for model, data in zip(modelsR, data_X)]

    print(predictions)
    df = pd.DataFrame([predictions])
    df.to_csv("predictions_stuff.csv", mode='a', header=False, index=False)

    binary_predictions = [[1 if pred >= 0.5 else 0 for pred in prediction] for prediction in predictions]

    current_diagnosis = []

    for i, binary_prediction in enumerate(binary_predictions):
        print(f"Predictions for model {labels[i+1]}: {binary_prediction}")
        current_diagnosis.append((labels[i+1], binary_prediction))

    diagnoseGloblins = [] # just storing a list of the names, and pruning results just in case...

    for diagnosis in current_diagnosis:
        if diagnosis[1][0] == 1:
            diagnoseGloblins.append(diagnosis[0])

    if len(diagnoseGloblins) != 0:
        if not "kappa" in diagnoseGloblins and not "lambda" in diagnoseGloblins:
            if X_K[0] < X_L[0] and X_K[0] != 0: #check make sure value isn't 0 cuz that could be an error
                diagnoseGloblins.append("kappa")
            else :
                # print(X_L[0], X_K[0], data[data['Label'] == "lambda"][['Brightness']].iloc[i:]['Brightness'].min(), data[data['Label'] == "kappa"][['Brightness']].iloc[i:]['Brightness'].min())
                diagnoseGloblins.append("lambda")

    if len(diagnoseGloblins) < 2 or len(diagnoseGloblins) > 2: # make sure there is only a pair of values, if not, nomo
        diagnoseGloblins = []
    
    print(brightness_values_saved)
    if "g" in diagnoseGloblins and "lambda" in diagnoseGloblins or "g" in diagnoseGloblins and "kappa" in diagnoseGloblins:
        if -60 < brightness_values_saved[-2] - brightness_values_saved[-1] < 150:
            diagnoseGloblins = []
    if "a" in diagnoseGloblins and "lambda" in diagnoseGloblins:
        if -30 < brightness_values_saved[2] - brightness_values_saved[-1] < 25:
            diagnoseGloblins = []
    if "a" in diagnoseGloblins and "lambda" in diagnoseGloblins:
        if brightness_values_saved[2] - brightness_values_saved[-1] < -101:
            diagnoseGloblins = ['a', 'kappa']
    if "m" in diagnoseGloblins and "kappa" in diagnoseGloblins:
        if -2 < brightness_values_saved[3] - brightness_values_saved[-2] < 5:
            diagnoseGloblins = []
    if "m" in diagnoseGloblins and "lambda" in diagnoseGloblins:
        if 15 < brightness_values_saved[3] - brightness_values_saved[-1]:
            diagnoseGloblins = []
                
    return diagnoseGloblins