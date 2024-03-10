import pandas as pd

labels = ["g", "a", "m", "kappa", "lambda"] # SP isn't included cuz its not needed

diagnosis = [ # gamlo detected, gamlo value if outside of reference range
    [11, "GL", 0, 0],
    [13, "NOMO", 0, 0],
    [14, "NOMO", 1, 0.37],
    [15, "MK", 0, 0],
    [17, "GL", 0, 0],
    [18, "AL", 0, 0],
    [20, "NOMO", 1, 0.41],
    [22, "GK", 1, 0.58],
    [24, "NOMO", 0, 0.70],
    # [25, "NOMO"], hemol
    [27, "NOMO", 1, 0.49],
    [32, "GK", 0, 1.93],
    [34, "NOMO", 0, 1.77],
    [35, "NOMO", 0, 0],
    [37, "GK", 0, 2.10],
    [40, "NOMO", 1, 0.56],
    [41, "AL", 1, 0.53],
    [42, "NOMO", 0, 0],
    [44, "AK", 0, 0.64],
    [45, "GL", 0, 0],
    [46, "GL", 1, 0.58],
    [47, "NOMO", 0, 0],
    [49, "MK", 0, 0],
    [50, "NOMO", 0, 0],
    [51, "NOMO", 1, 0.61],
    [52, "NOMO", 0, 0],
    [54, "NOMO", 0, 0],
    [55, "NOMO", 0, 0],
    [58, "NOMO", 0, 0],
    [59, "NOMO", 0, 0],
    [62, "NOMO", 0, 0],
    [63, "NOMO", 0, 0.68],
    [64, "NOMO", 0, 0],
    [65, "NOMO", 0, 0],
    [67, "NOMO", 1, 0.29],
    [69, "NOMO", 0, 0.70],
    [70, "NOMO", 0, 1.89],
    # [71, "NOMO"], hemol
    [72, "AK", 0, 0.47],
    [73, "GL", 0, 0],
    [74, "NOMO", 0, 0],
    [76, "NOMO", 0, 0.78],
    [77, "GK", 1, 0.59],
    # [78, "FGK"],
    # [79, "GK"], hemol
    [80, "GK", 0, 0],
    [81, "NOMO", 1, 0.22],
    [84, "GL", 0, 1.88],
    [85, "GK", 0, 0],
    [86, "NOMO", 0, 0],
    [87, "GK", 0, 0],
    # [88, "NOMO", ],
    [92, "NOMO", 0, 0],
    [101, "NOMO", 0, 0],
    [102, "NOMO", 0, 0.72],
    [103, "NOMO", 0, 0],
    [105, "NOMO", 1, 0.34],
    [107, "GK", 0, 2.13],
    [108, "AL", 0, 0],
    [109, "NOMO", 0, 0],
    [110, "NOMO", 1, 0.52],
    [112, "GL", 0, 0],
    # [113, "NOMO", ],
    [116, "NOMO", 0, 0.69],
    [117, "GK", 0, 1.73],
    [118, "GK", 0, 0],
    [120, "NOMO", 1, 0.14],
    [122, "NOMO", 0, 0],
    [123, "GL", 0, 0],
    [125, "NOMO", 1, 0.49],
    [126, "GK", 0, 0],
    # [132, "GL", ],
    [134, "NOMO", 0, 0.75],
    [135, "NOMO", 0, 0],
    [136, "NOMO", 0, 0],
    [137, "NOMO", 1, 0.63],
    [138, "NOMO", 0, 0],
    [141, "AK", 0, 3.88],
    [144, "GL", 0, 1.93],
    [145, "NOMO", 0, 0],
    [149, "NOMO", 1, 0.58],
    [150, "NOMO", 0, 0.71],
    [152, "GL", 0, 1.85],
    [153, "AK", 0, 0.70],
    # [157, "GK", 0, 0.],
    [158, "NOMO", 0, 0],
    [159, "NOMO", 0, 0],
    [160, "NOMO", 1, 0.26],
    [162, "NOMO", 0, 0.76],
    [163, "NOMO", 0, 0],
    [164, "NOMO", 0, 0],
    [165, "NOMO", 0, 0],
    [166, "NOMO", 0, 0],
    [168, "NOMO", 0, 0],
    [169, "NOMO", 0, 0],
    [170, "GK", 0, 0.78],
    [172, "NOMO", 0, 0],
    [174, "NOMO", 0, 0.64],
    [175, "NOMO", 0, 0],
    [177, "NOMO", 0, 0],
    [178, "NOMO", 0, 0.69],
    [179, "GK", 1, 0.39],
    [180, "GK", 0, 1.83],
    [181, "AK", 1, 0.22],
    [183, "MK", 0, 0],
    # [186, "NOMO", ],
    [188, "NOMO", 0, 0],
    [192, "NOMO", 1, 0.44],
    [193, "GK", 0, 0],
    [196, "NOMO", 0, 0],
    [198, "NOMO", 0, 0],
    [201, "GL", 0, 0],
    [203, "NOMO", 0, 0.78],
    [204, "GK", 0, 0],
    [209, "GK", 0, 0],
    [210, "NOMO", 1, 0.50],
    [223, "NOMO", 0, 0.67],
    [225, "NOMO", 1, 0.60],
    [226, "NOMO", 1, 0.61],
    [227, "AK", 0, 0.65],
    [228, "GK", 0, 0],
    [229, "GK", 0, 1.94],
    [230, "NOMO", 1, 0.63],
    [232, "ML", 0, 3.27],
    [233, "GK", 0, 4.18],
    [234, "NOMO", 0, 1.84],
    [236, "NOMO", 0, 0],
    [239, "ML", 1, 0.24],
    [241, "GK", 0, 0.76],
    # [242, "NOMO", ],
    [245, "NOMO", 0, 0],
    [246, "NOMO", 0, 0],
    [249, "NOMO", 0, 0],
    [250, "GK", 0, 0],
    [251, "NOMO", 1, 0.40],
    [252, "NOMO", 0, 0],
    [253, "NOMO", 0, 0],
    [254, "NOMO", 0, 0.68],
    [256, "NOMO", 0, 0],
    [257, "GL", 0, 0],
    [259, "NOMO", 0, 0],
    [260, "NOMO", 0, 0],
    [261, "NOMO", 0, 0],
    [262, "NOMO", 0, 0],
    [263, "NOMO", 0, 0],
    [264, "NOMO", 0, 0],
    [266, "NOMO", 0, 0],
    [267, "NOMO", 0, 0],
    [268, "NOMO", 0, 0.74],
    [269, "NOMO", 0, 0],
    [270, "NOMO", 0, 0],
    [271, "NOMO", 0, 0],
    [272, "NOMO", 0, 0],
    [273, "NOMO", 0, 0],
    [274, "NOMO", 1, 0.45],
    [275, "NOMO", 0, 0],
    [277, "NOMO", 0, 3.90],
    [279, "NOMO", 0, 0.67],
    [282, "NOMO", 0, 0],
    [284, "NOMO", 0, 0],
    [288, "AL", 0, 0],
    [289, "NOMO", 0, 2.26],
    # [290, "NOMO", ],
    [292, "NOMO", 1, 0.38],
    [293, "AL", 0, 0.59],
    [296, "NOMO", 1, 0.27],
    [299, "GK", 0, 0],
    [300, "NOMO", 0, 0.72],
    [301, "GL", 0, 0],
    [302, "NOMO", 0, 0.72],
    [303, "NOMO", 0, 0],
    [304, "NOMO", 0, 0.65],
    [306, "NOMO", 0, 0.42],
    [307, "NOMO", 1, 0.50],
    # [309, "GK", ],
    # [310, "GK", ],
    [311, "NOMO", 0, 0],
    [312, "NOMO", 0, 0.75],
    [314, "NOMO", 0, 0],
    [315, "GK", 0, 1.90],
    [316, "GL", 0, 0],
    [324, "NOMO", 0, 0],
    [325, "NOMO", 0, 0],
    # [326, "GK", ],
    [327, "NOMO", 0, 0.68],
    [328, "NOMO", 0, 0],
    [329, "GK", 0, 3.87],
    [331, "MK", 0, 0.72],
    [333, "GK", 1, 0.15],
    [334, "MK", 0, 0],
    [335, "NOMO", 0, 0],
    [337, "GK", 0, 3.97],
    [339, "NOMO", 0, 0.64],
    [340, "NOMO", 0, 0],
    [341, "NOMO", 1, 0.31],
    [342, "NOMO", 1, 0.58],
    [343, "NOMO", 0, 0],
    [344, "NOMO", 1, 0.43],
    [345, "NOMO", 1, 0.27],
    [346, "NOMO", 0, 0],
    [348, "GK", 0, 0],
    [349, "AK", 0, 0],
    [350, "NOMO", 1, 0.33],
    [351, "GK", 0, 0],
    [355, "GL", 0, 1.94],
    [359, "GL", 0, 0],
    [361, "GK", 1, 0.57],
    [366, "GL", 0, 0],
    [369, "GK", 1, 0.61],
    [370, "AK", 1, 0.53],
    [372, "GK", 0, 0],
    [374, "NOMO", 0, 0],
    [376, "NOMO", 1, 0.4],
    [379, "NOMO", 0, 0],
    [381, "AK", 0, 1.95],
    [382, "GK", 0, 0],
    [383, "NOMO", 1, 0.31],
    [384, "NOMO", 1, 0.24],
    [385, "AL", 1, 0.58],
    [386, "NOMO", 0, 0],
    [387, "NOMO", 0, 0],
    [388, "NOMO", 0, 0],
    [389, "NOMO", 0, 0],
    [390, "NOMO", 0, 4.16],
    [391, "NOMO", 0, 0],
    [392, "NOMO", 0, 0],
    [393, "NOMO", 0, 0],
    [395, "NOMO", 0, 2.40],
    [396, "NOMO", 0, 1.75],
    [397, "NOMO", 0, 0.24],
    [398, "NOMO", 0, 0],
    [399, "NOMO", 0, 0],
    [400, "NOMO", 1, 0.57],
    [401, "NOMO", 0, 0],
    [402, "NOMO", 0, 0],
    [403, "NOMO", 0, 0],
    [404, "NOMO", 0, 1.91],
    [406, "NOMO", 0, 0],
    [407, "NOMO", 0, 0],
    [408, "NOMO", 1, 0.59],
    [409, "NOMO", 1, 0.51],
    [410, "NOMO", 1, 0.60],
    [411, "NOMO", 0, 1.90],
    [413, "NOMO", 0, 0.67],
    [414, 'NOMO', 0, 0.77],
    [415, "NOMO", 0, 0],
    [416, "NOMO", 0, 0],
    [418, "NOMO", 0, 0],
    [419, "NOMO", 0, 0],
    [420, "NOMO", 0, 0.73],
    [423, "NOMO", 0, 0],
    [424, "NOMO", 1, 0.61],
    [426, "NOMO", 0, 0],
    [428, "NOMO", 0, 0.64],
    [431, "NOMO", 1, 0.54],
    [433, "NOMO", 1, 0.26],
    [435, "NOMO", 1, 0.24],
    [440, "NOMO", 0, 0],
    [442, "NOMO", 0, 0.18],
    [445, "NOMO", 0, 0],
    [450, "NOMO", 1, 0.42],
    [452, "GL", 0, 0],
    [454, "NOMO", 0, 2.35], #poly
    [455, "NOMO", 0, 2.79], #poly
    [456, "NOMO", 0, 0],
    [458, "NOMO", 0, 0],
    [459, "NOMO", 0, 0],
    [460, "NOMO", 0, 0],
    [461, "NOMO", 0, 0],
    [463, "NOMO", 0, 0],
    [465, "NOMO", 0, 0],
    [466, "NOMO", 0, 0],
    [467, "NOMO", 0, 0],
    [468, "NOMO", 0, 0.79],
    [470, "NOMO", 0, 0],
    [471, "NOMO", 0, 0],
    [472, "NOMO", 0, 0.75],
    [474, 'NOMO', 0, 0],
    [475, "NOMO", 0, 1.91],
    [476, "NOMO", 0, 1.73],
    [479, "NOMO", 0, 0.35],
    [480, "GL", 0, 0],
    [481, "NOMO", 0, 0],
    [482, "NOMO", 0, 0.74],
    [483, "NOMO", 0, 0],
    [489, "GL", 0, 0],
    [492, "NOMO", 0, 0.73],
    [493, "NOMO", 0, 0.79],
    [494, "NOMO", 0, 0],
    [495, "NOMO", 0, 0],
    [498, "NOMO", 0, 0.45],
    [501, "NOMO", 0, 0.55],
    [502, "GL", 0, 0],
    [507, "MK", 0, 0],
    [517, "GK", 0, 1.97],
    [518, "GK", 0, 1.91],
    [526, "GL", 0, 0],
    [528, "GL", 0, 0.67],
    [529, "NOMO", 0, 0.75],
    # [531, "NOMO", 0, 0], # outlier i think this might sscrew things up, its fine if i get it wrong cuz theres still a ton of other data
    [533, "NOMO", 0, 0.43],
    [535, "NOMO", 0, 0],
    [536, "NOMO", 0, 0.62],
    [538, "GL", 0, 2.88],
    [544, "AK", 0, 2.10],
    [545, "MK", 0, 1.81],
    [547, "GK", 0, 0.55],
    [548, "NOMO", 0, 0.36],
    [549, "GK", 0, 0],
    [552, "NOMO", 0, 0],
    [554, "GK", 0, 2.19],
    [556, "GK", 0, 4.22],
    [560, "GL", 0, 1.96],
    [563, "NOMO", 0, 2.07], #poly
    [564, "GK", 0, 0],
    [568, "NOMO", 0, 2.55],
    [570, "GK", 0, 2.38],
    [572, "NOMO", 0, 2.03],
    [573, "GL", 0, 2.25],
    [575, "NOMO", 0, 1.71],
    [577, "NOMO", 0, 0],
    [578, "NOMO", 0, 0],
    [580, "NOMO", 0, 0],
    [585, "NOMO", 0, 0],
    [587, "NOMO", 0, 0], 

    [29, "NOMO", 0, 1.31],
    [38, "NOMO", 0, 1.55],
    [61, "NOMO", 0, 1.40],
    [68, "NOMO", 0, 1.39],
    [176, "NOMO", 0, 1.50],
    [247, "NOMO", 0, 1.36],
    [286, "NOMO", 0, 1.50],
    [304, "NOMO", 1, 0.66],
    [457, "NOMO", 0, 1.40],
    [485, "NOMO", 0, 1.31],
    [513, "NOMO", 0, 1.27],
    [571, "NOMO", 0, 1.33],
    [38, "NOMO", 0, 1.55],
    [61, "NOMO", 0, 1.40],
    [68, "NOMO", 0, 1.39],
    [176, "NOMO", 0, 1.50],
    [247, "NOMO", 0, 1.36],
    [286, "NOMO", 0, 1.50],
    [304, "NOMO", 1, 0.66],
    [457, "NOMO", 0, 1.40],
    [485, "NOMO", 0, 1.31],
    [513, "NOMO", 0, 1.27],
    [571, "NOMO", 0, 1.33],
    [38, "NOMO", 0, 1.55],
    [61, "NOMO", 0, 1.40],
    [68, "NOMO", 0, 1.39],
]

from sklearn.utils import resample

df = pd.DataFrame(diagnosis, columns=['Value', 'Class', 'gamlodetected', 'gammavalue'])

# resample data to the average
average = int(df['Class'].value_counts().mean())

max_count = df['Class'].value_counts().max()

resampled_df = pd.DataFrame()

# Resample each class to equal the maximum count
for class_name in df['Class'].unique():
    class_df = df[df['Class'] == class_name]
    resampled_class_df = resample(class_df, replace=True, n_samples=max_count, random_state=42)
    resampled_df = pd.concat([resampled_df, resampled_class_df])

diagnosis = resampled_df.values.tolist()
print(resampled_df)

# create seperate arrays for each of the different data values depending on its labels
X_G = [] 
X_A = []
X_M = []
X_K = []
X_L = []
y_g = [] # 0 or 1, whether it is it's label, e.g. "lambda", or its not
y_a = []
y_m = []
y_k = []
y_l = []

# step 1, open the file
for x in range(len(diagnosis)):
    data = pd.read_csv(f"peaks/brightness_averages_{diagnosis[x][0]}.csv")

    # step 2, loop through each string in the array and based on the string, this will determine what values are taken into account for the neural network
    values_to_take_into_account = []
    for y in range(len(labels)):
        if labels[y] in diagnosis[x][1].lower() and diagnosis[x][1] != "NOMO" or 'k' in diagnosis[x][1].lower() and y == 3 or 'l' in diagnosis[x][1].lower() and y == 4:
            values_to_take_into_account.append(labels[y])
            # print(diagnosis[x][1], values_to_take_into_account)
            # print(diagnosis[x][1], values_to_take_into_account)
        elif diagnosis[x][1]=="NOMO":
            break
        else:
            values_to_take_into_account.append("")
            
    # step 2.1 extract data from the columns with the values_to_take_into_account
    for value in labels:
        # print(value, value in values_to_take_into_account)
        if value in values_to_take_into_account:
            extracted_data = data[data['Label'] == value][['Brightness']]

            # step 2.2 with the extracted data, find the min and max
            min_brightness_values = extracted_data['Brightness'].min()   
            max_brightness_values = extracted_data['Brightness'].max()   

            # step 2.3 added data to array & normalize it
            if value == "g":
                X_G.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_g.append(1)
            elif value == "a":
                X_A.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_a.append(1)
            elif value == "m":
                X_M.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_m.append(1)
            elif value == "kappa":
                X_K.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_k.append(1)
            elif value == "lambda":
                X_L.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_l.append(1)
        else:
            extracted_data = data[data['Label'] == value][['Brightness']]

            # step 2.2 with the extracted data
            min_brightness_values = extracted_data['Brightness'].min()   
            max_brightness_values = extracted_data['Brightness'].max()   

            # For all the values that weren't taken into account, add a 0 to the y
            if value == "g":
                X_G.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_g.append(0)
            elif value == "a":
                X_A.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_a.append(0)
            elif value == "m":
                X_M.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_m.append(0)
            elif value == "kappa":
                X_K.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_k.append(0)
            elif value == "lambda":
                X_L.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                y_l.append(0)
            max_brightness_values = 255

            if min_brightness_values == 0: # fixing messed up data
                if value == "g":
                    X_G.append((round(1,7), diagnosis[x][3]))
                    y_g.append(0)
                elif value == "a":
                    X_A.append((round(1,7), diagnosis[x][3]))
                    y_a.append(0)
                elif value == "m":
                    X_M.append((round(1,7), diagnosis[x][3]))
                    y_m.append(0)
                elif value == "kappa":
                    X_K.append((round(1,7), diagnosis[x][3]))
                    y_k.append(0)
                elif value == "lambda":
                    X_L.append((round(1,7), diagnosis[x][3]))
                    y_l.append(0)
            else:

                # For all the values that weren't taken into account, add a 0 to the y
                if value == "g":
                    X_G.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                    y_g.append(0)
                elif value == "a":
                    X_A.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                    y_a.append(0)
                elif value == "m":
                    X_M.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                    y_m.append(0)
                elif value == "kappa":
                    X_K.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                    y_k.append(0)
                elif value == "lambda":
                    X_L.append((round(min_brightness_values/max_brightness_values,7), diagnosis[x][3]))
                    y_l.append(0)


diagnosis_values = [diagnosis[i][0] for i in range(len(diagnosis))]

# step 3, make a seperate individual neural network for each label, "g", "a", "m" etc.
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
from sklearn.model_selection import train_test_split

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.005)

def create_model():
    model = Sequential()
    model.add(Dense(2, activation='relu', input_shape=(2,)))
    model.add(BatchNormalization())
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='sigmoid', input_shape=(1,)))
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a neural network for each array
model_X_G = create_model()
model_X_A = create_model()
model_X_M = create_model()
model_X_K = create_model()
model_X_L = create_model()

# convert the arrays into numpy arrays
X_G = np.array(X_G)
X_A = np.array(X_A)
X_M = np.array(X_M)
X_K = np.array(X_K)
X_L = np.array(X_L)

y_g = np.array(y_g)
y_a = np.array(y_a)
y_m = np.array(y_m)
y_k = np.array(y_k)
y_l = np.array(y_l)

# split the data into training and test sets
X_train_G, X_test_G, y_train_G, y_test_G = train_test_split(X_G, y_g, test_size=0.5, random_state=42)
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(X_A, y_a, test_size=0.5, random_state=42)
X_train_M, X_test_M, y_train_M, y_test_M = train_test_split(X_M, y_m, test_size=0.5, random_state=42)
X_train_K, X_test_K, y_train_K, y_test_K = train_test_split(X_K, y_k, test_size=0.5, random_state=42)
X_train_L, X_test_L, y_train_L, y_test_L = train_test_split(X_L, y_l, test_size=0.5, random_state=42)

epochs_amount = 64
batch_size = 64

# Fit the models and save the histories to visualize the graph later
history_X_G = model_X_G.fit(X_train_G, y_train_G, epochs=epochs_amount, batch_size=batch_size, verbose=0) 
history_X_A = model_X_A.fit(X_train_A, y_train_A, epochs=epochs_amount, batch_size=batch_size, verbose=0)
history_X_M = model_X_M.fit(X_train_M, y_train_M, epochs=epochs_amount, batch_size=batch_size, verbose=0)
history_X_K = model_X_K.fit(X_train_K, y_train_K, epochs=epochs_amount, batch_size=batch_size, verbose=0)
history_X_L = model_X_L.fit(X_train_L, y_train_L, epochs=epochs_amount, batch_size=batch_size, verbose=0)

# Evaluate the models on the test data
score_X_G = model_X_G.evaluate(X_test_G, y_test_G, verbose=0)
score_X_A = model_X_A.evaluate(X_test_A, y_test_A, verbose=0)
score_X_M = model_X_M.evaluate(X_test_M, y_test_M, verbose=0)
score_X_K = model_X_K.evaluate(X_test_K, y_test_K, verbose=0)
score_X_L = model_X_L.evaluate(X_test_L, y_test_L, verbose=0)

print(f"Accuracy of model_X_G on test data: {score_X_G[1]*100}%")
print(f"Accuracy of model_X_A on test data: {score_X_A[1]*100}%")
print(f"Accuracy of model_X_M on test data: {score_X_M[1]*100}%")
print(f"Accuracy of model_X_K on test data: {score_X_K[1]*100}%")
print(f"Accuracy of model_X_L on test data: {score_X_L[1]*100}%")

# step 4 visualize the trends of learning to determine if overfitting might be occuring
import matplotlib.pyplot as plt

# Plot the training accuracy of each model
plt.figure(figsize=(12, 6))
plt.plot(history_X_G.history['accuracy'])
plt.plot(history_X_A.history['accuracy'])
plt.plot(history_X_M.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train_X_G', 'Train_X_A', 'Train_X_M', 'Train_X_K', 'Train_X_L'], loc='upper left')
plt.show()


# step 4.1 save the model. md for more data, a for the test size update
# model_X_G.save(f'models/model_X_G+{epochs_amount}+{batch_size}.keras')
# model_X_A.save(f'models/model_X_A+{epochs_amount}+{batch_size}.keras')
# model_X_M.save(f'models/model_X_M+{epochs_amount}+{batch_size}.keras')
# model_X_K.save(f'models/model_X_K+{epochs_amount}+{batch_size}.keras')
# model_X_L.save(f'models/model_X_L+{epochs_amount}+{batch_size}.keras')