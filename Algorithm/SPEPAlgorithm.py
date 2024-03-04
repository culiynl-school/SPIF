import cv2
from CropImage import *
import numpy as np

labels = ['albumin', 'alpha-1', 'alpha-2', 'beta', 'gamma']

lower = np.array([200, 180, 150])  # example for black
upper = np.array([255, 200, 180])  # example for near black

def find_contour_points(segment):
    # find the contour points
    height, width = segment.shape[:2]
    # crop the image to remove first 3px and last 3px due to lines.
    cropAmnt = round(0.05*width)
    segment = crop_image(segment, 0, height, cropAmnt, width-cropAmnt)

    # Convert the image to grayscale and binarize it
    gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize lists to store x and y coordinates
    x_coords = []
    y_coords = []

    # Extract x and y coordinates from each contour point
    for contour in contours:
        for point in contour:
            x_coords.append(point[0][0])
            y_coords.append(point[0][1])

    # Combine the lists into a list of tuples
    coords = list(zip(x_coords, y_coords))

    coords = list(dict(coords).items())

    # Sort the list of tuples
    coords.sort()

    # Unzip the sorted list into two lists
    x_coords, y_coords = zip(*coords)

    return x_coords, y_coords

def SPEPAlgorithm(img):
    diagnosis = []
    image = cv2.imread("PDFScan/sample"+str(img)+".jpg") 

    image = crop_image(image, 136, 285, 33, 269)

    image = trim_white_space(image) 

    height, width = image.shape[:2] 

    padding = 3
    image = crop_image(image, round(padding), round(height-padding), round(padding*2.7), round(width-padding)) # crop the image to remove some of the lines found on the sides
    image = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)

    # identify black lines to segment the image
    
    x_coordinates = []
    y_coordinates = []
    
    segment_width = width // 40

    height, width = image.shape[:2]

    for i in range(0, width, segment_width):  # Loop over x-coordinates (columns) with step size of segment_width
        # Define the range of y-coordinates for the column
        y_range = range(int(height*0.8), height)

        # find the average column color
        column_color = image[y_range, i]
        
        # Ignore values > 250
        column_color = column_color[np.where(column_color <= 220)]

        # Calculate the average color
        average_color = np.average(column_color, axis=0)
        average_color = np.average(average_color)

        if 30 < average_color < 120:
            x_coordinates.append(i)
            y_coordinates.append(height*0.9) 
    
    for i in range(image.shape[1]):  # Loop over x-coordinates (columns)
        for j in range(int(height*0.8), image.shape[0]):  # Loop over y-coordinates (rows)
            # Check if the color values are within a distance of 10 from the average color value
            averageImageColor = np.average(image[j, i])

            height, width = image.shape[:2] 
            if abs(round(image[j, i, 0]) - round(image[j, i, 1])) <= 1 and abs(round(image[j, i, 0]) - round(image[j, i, 2])) <= 1 and averageImageColor <= 160:
                too_close = False  # Variable to check if the point is too close to another point
                for x in x_coordinates:
                    if abs(i - x) < 50:  # If the point is too close to another point
                        too_close = True
                        break
                if not too_close and i > 25 and j > (height*0.8):
                    if len(x_coordinates) == 4: # checking for last point just in case it accidentally gets the yellow lines
                        if i > 660:
                            x_coordinates.append(i)
                            y_coordinates.append(j)
                    else :
                        x_coordinates.append(i)
                        y_coordinates.append(j)
                break

    # sort x coords and y coords
    points = list(zip(x_coordinates, y_coordinates))

    # Remove duplicates and sort by x-coordinate
    points = sorted(set(points), key=lambda x: x[0])

    # Unzip the list of tuples back into two lists
    x_coordinates, y_coordinates = zip(*points)

    x_coordinates = list(x_coordinates)
    y_coordinates = list(y_coordinates)

    while len(x_coordinates) > 5:
        print("Removing extra points since it is unneeded")
        x_coordinates.pop()
        y_coordinates.pop()

    if len(x_coordinates) < 5:
        print("Missing a point! Attempting to add a point...")

    # The default coordinates you can choose from
    x_default_coords = [170, 230, 350, 500, round(width*0.8)]
    
    def add_points(x_coordinates, y_coordinates, x_default_coords, height):
        while len(x_coordinates) < 5:
            min_differences = [min(abs(x - point) for x in x_coordinates) for point in x_default_coords]

            new_point = x_default_coords[min_differences.index(max(min_differences))]

            x_coordinates.append(new_point)
            y_coordinates.append(height * 0.9)

            x_default_coords.remove(new_point)

        x_coordinates.sort()

        return x_coordinates, y_coordinates

    x_coordinates, y_coordinates = add_points(x_coordinates, y_coordinates, x_default_coords, height)
    # print(x_coordinates)
    
    # fig, ax = plt.subplots()
    # ax.imshow(image)
    # ax.plot(x_coordinates, y_coordinates, 'ro')
    # plt.show()

    # segment the image
    segmented_images = []
    segmented_images.append(image[:, 0:x_coordinates[0]])

    # Segment the image based on the x-values
    for i in range(len(x_coordinates) - 2):
        segment = image[:, x_coordinates[i]:x_coordinates[i+1]]
        
        segmented_images.append(segment)

    # some extra stuff to get the last line at the end
    if len(x_coordinates) == 5:
        segmented_images.append(image[:, x_coordinates[len(x_coordinates)-2]:x_coordinates[len(x_coordinates)-1]])
    else: 
        segmented_images.append(image[:, x_coordinates[-1]:])

    labeled_images = dict(zip(labels, segmented_images))

    # scan the image to find contour points

    colorValuesX = [195, 225]
    colorValuesY = [170, 195]
    colorValuesZ = [130, 170]

    areas = []

    min_mask = 160

    # loop through each segmented image
    for i, (label, segment) in enumerate(labeled_images.items()):
        
        height, width = segment.shape[:2] 
        # crop the image to remove first 3px and last 3px due to lines. 
        cropAmnt = 7
        segment = crop_image(segment, 0, height, cropAmnt, width-cropAmnt)

        # Convert the image to grayscale and binarize it
        gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, min_mask, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize lists to store x and y coordinates
        x_coords = []
        y_coords = []

        # Extract x and y coordinates from each contour point
        for contour in contours:
            for point in contour:
                x_coords.append(point[0][0])
                y_coords.append(point[0][1])
        
        # plt.imshow(binary)
        # plt.plot(x_coords, y_coords)
        # plt.show()

        # Convert lists to numpy arrays
        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

        coords = list(zip(x_coords, y_coords))

        coords = list(dict(coords).items())

        # Sort the list of tuples
        coords.sort()

        # Unzip the sorted list into two lists
        x_coords, y_coords = zip(*coords)

        area = 0
        for i in range(len(x_coords)-1):
            area += (x_coords[i+1] - x_coords[i]) * (y_coords[i+1] + y_coords[i]) / 2

        areas.append((area, label))
        
        # print(y_coords)

        # print(np.min(y_coords), label)

        peakDecNum = 457 # this number corresponds to the y value that the min_contour_point
        # this is a fall back, meaning that if this runs, then that means that is **absolutely has to be** an mspike since it is way off the charts, unless everything is already off the charts
        # print(np.min(y_coords))
        if label == 'gamma' and np.min(y_coords) <= peakDecNum or label == 'beta' and np.min(y_coords) <= peakDecNum*0.8:
            diagnosis.append(['MSP', label])
            mspikeFound = True
            print("MSP detected in " + label)
        else:
            diagnosis.append(('MSND',label))

        # diagnosis = "MSND"

        ###### HANDLING THE YELLOW LINES ######

        contours = []
        
        x_coordinates_lines = []
        y_coordinates_lines = []
        # find the auto generated lines
        for i in range(segment.shape[1]):
            for j in range(segment.shape[0]):

                colorValuesX = [195, 225]
                colorValuesY = [170, 195]
                colorValuesZ = [130, 170]

                if colorValuesX[0] < segment[j][i][0] < colorValuesX[1] and colorValuesY[0] < segment[j][i][1] < colorValuesY[1] and colorValuesZ[0] < segment[j][i][2] < colorValuesZ[1]:
                    too_close = False
                    for x in x_coordinates_lines:
                        if abs(i - x) < 25:
                            too_close = True
                            break
                    if not too_close:
                        x_coordinates_lines.append(i)
                        y_coordinates_lines.append(j)
                    break

        if len(x_coordinates_lines) == 1:
            x_coordinates_lines.append(1)
            x_coordinates_lines.append(width-1)

        
        # plt.scatter(x_coordinates_lines, y_coordinates_lines)
        # plt.imshow(segment)
        # plt.show()
        
        # now, for each segment, segment the image again based on the lines if applicable
        if len(x_coordinates_lines) != 0:
            # After finding the points, segment the image again
            segmented_images_lines = []

            segmented_images_lines.append(segment[:, :x_coordinates_lines[0]])
            for i in range(len(x_coordinates_lines) - 1):
                # Extract the segment
                segmentR = segment[:, x_coordinates_lines[i]:x_coordinates_lines[i+1]]
                
                # Append the segment to the list
                segmented_images_lines.append(segmentR)
            segmented_images_lines.append(segment[:, x_coordinates_lines[-1]:])

            # Define lower and upper bounds for the colors you want to ignore
            lower = np.array([200, 180, 150])  # example for black
            upper = np.array([255, 200, 180])  # example for near black

            # print(len(segmented_images_lines))
            areaD = []
            for segment in segmented_images_lines:
                height, width = segment.shape[:2]
                # crop the image to remove first 3px and last 3px due to lines.
                cropAmnt = round(0.05*width)
                segment = crop_image(segment, 0, height, cropAmnt, width-cropAmnt)

                # preprocess
                mask = cv2.inRange(segment, lower, upper)
                segment = cv2.bitwise_and(segment, segment, mask=255-mask)

                gray = cv2.cvtColor(segment, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, min_mask, 255, cv2.THRESH_BINARY_INV)

                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                x_coords = []
                y_coords = []

                for contour in contours:
                    for point in contour:
                        x_coords.append(point[0][0])
                        y_coords.append(point[0][1])
                
                x_coords = x_coords[7:len(x_coords)-3] # cutting off some stuff because there is still some times it gets the spike peak

                coords = list(zip(x_coords, y_coords))

                coords = list(dict(coords).items())

                coords.sort()

                if coords:  # Check if coords is not empty
                    x_coords, y_coords = zip(*coords)
                else:
                    x_coords, y_coords = [], []
                        
                # plt.imshow(binary)
                # plt.plot(x_coords, y_coords)
                # plt.show()

                # after finding the contour points, find the area under the curve
                area = 0
                for i in range(len(x_coords)-1):
                    area += (x_coords[i+1] - x_coords[i]) * (y_coords[i+1] + y_coords[i]) / 2 # gauss's formula

                areaD.append(area)

            areas.append((np.sum(areaD), label))

            print(areaD[1] / np.sum(areaD), ((areaD[0] + areaD[2]) / np.sum(areaD)))
            mspikeFound = False
            # the 0 in diagnosis means that it is from the AUC calculation
            if label == "gamma":
                if 0.304 < areaD[1] / np.sum(areaD):
                    diagnosis.append(["MSP", 0])
                    mspikeFound = True
                else:
                    diagnosis.append(["MSND", 0])
            elif label == "beta":
                if 0.304 < areaD[1] / np.sum(areaD):
                    diagnosis.append(["MSP", "beta"])
                    mspikeFound = True
                else:
                    diagnosis.append(["MSND", 0])
            else:
                print("Yellow / blue lines were found somewhere else?")

    return diagnosis