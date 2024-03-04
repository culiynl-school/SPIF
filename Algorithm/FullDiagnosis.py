import cv2
import matplotlib.pyplot as plt
import sys
from CropImage import *
import time
from Tesseract import *
from SPEPAlgorithm import *
from IFENeuralNetwork import *
import re

def FullDiagnosis(source):
    currentDiagnosis = []
    print("---------------------------------")
    
    #################################
    # step 0 -> determine scan type (urine / serum)
    # used for future improvements in the algorithm
    # in order to take both urine and serum cases
    # this step would be eliminated in laboratories
    #################################
    cropped_img = crop_image("PDFScan/sample"+str(source)+".jpg", 99, 99+26, 217, 217+111, use_path=True)
    upscaled_images = upscale_image(cropped_img)
    scanType = readImage(upscaled_images[1], "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ")
    print("This test is a: " + scanType)

    if "urine" in scanType.lower():
        return source, "urine"
    
    #################################
    # step 1 -> scan for g/dl text
    #################################
    cropped_img = crop_image("PDFScan/sample"+str(source)+".jpg", 303, 303+52, 173, 173+50, use_path=True)
    upscaled_images = upscale_image(cropped_img)

    gdLText = readImage(upscaled_images[1], "0123456789.") 

    #################################
    # step 1.1 -> error checking
    # sometimes, g/dl text is read wrong
    # using algorithm to replace abnormal g/dl numbers
    # with correct ones.
    #################################

    gdLText = gdLText.split()

    # loop through all the gdl text nums, and if there is a 9 in front of it e.g. 9.31, then change it to a 0.
    updated_numbers = []

    for gdl_text_num in gdLText:
        # Check if the GDL text number starts with '9.'
        if gdl_text_num.startswith('9.'):
            # Replace '9.' with '0.'
            updated_num = '0' + gdl_text_num[1:]
        else:
            updated_num = gdl_text_num

        updated_numbers.append(updated_num)
    
    gdLText = updated_numbers

    # loop through all the gdl text nums, and if there isn't a . in the current number and next number
    # then combine them together and add a . between them
    updated_numbers = []
    i = 0
    while i < len(gdLText):
        if '.' in gdLText[i] and i + 1 < len(gdLText) and '.' not in gdLText[i + 1]:
            updated_numbers.append((gdLText[i] + gdLText[i + 1]))
            i += 2
        else:
            updated_numbers.append((gdLText[i]))
            i += 1
    
    gdLText = updated_numbers

    
    #################################
    # Step 1.2
    # Formatting to float for computations
    #################################

    gdLTextNums = []
    if gdLText != False:
        # convert array to nums
        for z in range(len(gdLText)):
            try:
                # Try to convert the entire string to float
                gdLTextNums.append(float(gdLText[z]))
            except ValueError:
                print(f"\033[93m Cannot convert {gdLText[z]} to float. Removing unwanted characters... \033[0m")
                if any(c.isalpha() for c in gdLText[z]):
                    # Remove letters from the string
                    gdLText[z] = re.sub('[a-zA-Z]', '', gdLText[z])
                ### remove characters here ####
                if gdLText[z] == ".":
                    gdLTextNums.append(0)
                # if the gdl text has more than 2 periods, remove the last one
                elif gdLText[z].count(".") > 1:
                    gdLTextNums.append(float(gdLText[z][0:gdLText[z].rfind(".")]))
                else:
                    try:
                        # Try to convert the first 4 characters to float
                        gdLTextNums.append(float(gdLText[z][0:4]))
                    except ValueError:
                        print(f"\033[91m Cannot convert to float. Something is wrong with the text recognition. \033[0m")
                        quit()

        # only take the first 4 characters for gdl values, just in case it gets something like 1.444 instead of 1.44
        if len(gdLTextNums) > 0:
            gdLTextNums = [round(float(i), 2) for i in gdLTextNums]
            print(gdLTextNums)

    while len(gdLTextNums) <= 4: # just in case the list is too short
        gdLTextNums.append(0)


    print("PDFScan/sample"+str(source)+".jpg")
    print("g/dL: " + str(gdLText) + " \n")

    if (len(gdLText) == 0):
        print("\033[93m No GDL was found! \033[0m")
        quit()

    #################################
    # step 1.3 -> checking for abnormalities in g/dl
    # compares to reference ranges & identifies whether g/dl is normal or abnormal
    #################################
    gdLRange = [ # these are set ranges 
        [3.20, 5.30], #albumin
        [0.10, 0.40], #a1
        [0.50, 1.00], #a2
        [0.60, 1.20], #beta
        [0.80, 1.70], #gamma
        [6.40, 8.30] # total protein
    ]
    names = ["Albumin", "Alpha-1", "Alpha-2", "Beta", "Gamma"]

    gDlAnalysis = [] # 1 means greater, 2 means lower, 0 is good

    try:
        if len(gdLTextNums)!=0:
            for y in range(len(gdLTextNums)):
                if gdLTextNums[y] > gdLRange[y][1]:
                    print(f"{names[y]} is greater than g/dL range!")
                    gDlAnalysis.append(1)
                elif gdLTextNums[y] < gdLRange[y][0]:
                    print(f"{names[y]} is below the g/dL range!")
                    gDlAnalysis.append(2)
                else:
                    print(f"{names[y]} is good.")
                    gDlAnalysis.append(0)
            
            # checking fot the total protein
            if sum(gdLTextNums) > gdLRange[-1][1]:
                print(f"\033[91mTotal protein is greater than g/dL range! {sum(gdLTextNums)-gdLRange[-1][1]}\033[0m")
            elif sum(gdLTextNums) < gdLRange[-1][0]:
                print(f"\033[91mTotal protein is less than g/dL range! {round(gdLRange[-1][0]-sum(gdLTextNums), 2)}\033[0m")
            else:
                print("\033[93mTotal protein is good.\033[0m")
    except:
        print ("Error w/GDL")

    #################################
    # step 2 -> SPEP Analysis
    # returns whether a certain section on the graph is considered an "m-spike"
    #################################
    
    #### identify mspikes #####
    currentDiagnosis = SPEPAlgorithm(source)

    #################################
    # step 3 -> IFE Analysis
    # returns whether a certain section on the graph is considered an "m-spike"
    #################################

    # do ife scan
    if gDlAnalysis[-1] >= 1.25:
        IFEScan = IFENeuralNetwork(source, float(gdLTextNums[-1]))
    elif gDlAnalysis[-1] == 2 or gDlAnalysis[-1] == 1: # out of the reference ranges
        IFEScan = IFENeuralNetwork(source, float(gdLTextNums[-1]))
    else:
        IFEScan = IFENeuralNetwork(source, 0)

    if IFEScan == False: # there is no IFE to scan
        IFEScan = []

    #################################
    # step 4 -> condition checking & final diagnosis
    #################################
    gamloDetected = False
    polyDetected = False
    betamDetected = False
    mspikeDetected = False
    mifxDetected = False

    conditions = []
    try :
        if not mspikeDetected and gdLTextNums[-1] > 2:
            print("\033[92m POLY \033[0m")
            polyDetected = True
            conditions.append("poly")
        if not betamDetected and gdLTextNums[-1] < 0.64:
            print("\033[92m GAMLO \033[0m")
            gamloDetected = True
            conditions.append("gamlo")
        if gdLTextNums[0] < 2.56:
            print("\033[92m ALBLO \033[0m")
            conditions.append("alblo")
        elif 9 > gdLTextNums[0] > 6.36:
            print("\033[92m ALBHI \033[0m")
            conditions.append("albhi")
        if gdLTextNums[1] < 0.08:
            print("\033[92m A1LO \033[0m")
            conditions.append("a1lo")
        elif 8 > gdLTextNums[1] > 0.48:
            print("\033[92m A1HI \033[0m")
            conditions.append("a1hi")
        if gdLTextNums[2] < 0.40:
            print("\033[92m A2LO \033[0m")
            conditions.append("a2lo")
        elif gdLTextNums[2] > 1.20:
            print("\033[92m A2HI \033[0m")
            conditions.append("a2hi")
        if gdLTextNums[3] < 0.48 and not betamDetected:
            print("\033[92m BETALO \033[0m")
            conditions.append("betalo")
        elif gdLTextNums[3] > 1.44 and not betamDetected:
            print("\033[92m BETAHI \033[0m")
            conditions.append("betahi")

    except: print("Ignoring diagnosis w/gDL due to inaccurate reading")
    if betamDetected and gdLTextNums[-2] > 1.60: 
        print("\033[92m BETAM \033[0m")
        currentDiagnosis = "MSP"

    if not IFEScan == [] and currentDiagnosis == "MSND" and not mspikeDetected and not betamDetected:
            if gdLTextNums[-1] <= 1.4:
                currentDiagnosis = "MIFX"
                mifxDetected = True

    diagnosis = ""

    # Format the ife diagnosis into a string
    if "f" in IFEScan:
        mifxDetected = True
    IFEScan = ''.join([i.upper() if i == 'g' else 'K' if i == 'kappa' else 'L' if i == 'lambda' else i for i in IFEScan])

    if mifxDetected or mspikeDetected:
        if IFEScan != "":
            diagnosis = "MONO " + IFEScan.upper()

    if diagnosis == "" or not mspikeDetected and not mifxDetected:
        diagnosis = "NOMO"
        
    if currentDiagnosis == "MSP" and diagnosis == "NOMO":
        currentDiagnosis = "MSND UPDATE"
        mspikeDetected = False

    if currentDiagnosis == "MSND" and diagnosis != "NOMO":
        currentDiagnosis = "MSP UPDATE"
        mspikeDetected = True

    print("\033[92m" + str(currentDiagnosis) + "\033[0m")
    print(f"IFE Diagnosis: \033[92m{diagnosis}\033[0m")

    if not mspikeDetected and gdLTextNums[-1] > 2 and "poly" not in conditions:
        print("\033[92m POLY \033[0m")
        polyDetected = True
        conditions.append("poly")

    ## STEP 5 ##
    # ICD-10 CODES #
    if mspikeDetected or mifxDetected:
        print("\033[93m ICD-10 Code: D47.2 \033[0m")
    elif gamloDetected:
        print("\033[93m ICD-10 Code: D80.1 \033[0m")
    elif polyDetected:
        print("\033[93m ICD-10 Code: D89.0 \033[0m")
    else:
        print("\033[93m ICD-10 Code: E88.09 \033[0m")

    return source, currentDiagnosis, diagnosis, conditions

# wait for user input
while True:
    num = input("Enter the patient number you want to diagnose: ")
    FullDiagnosis(num)