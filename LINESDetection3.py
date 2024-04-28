import cv2 as cv
import numpy as np
import os
import csv





# folder_path = 'Microscope images'
folder_path = 'Re_ Microscope images'
# folder_path = 'TESTimages'
file_names = os.listdir(folder_path)
for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)

    img  = cv.imread(file_path)
    gray = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(gray, threshold1=0, threshold2=40)

    max_value = 255
    threshold_value =0
    _, binary_image1 = cv.threshold(gray, threshold_value, max_value, cv.THRESH_BINARY) #all of the pixels
    max_value = 255
    threshold_value = 100
    _, binary_image2 = cv.threshold(gray, threshold_value, max_value, cv.THRESH_BINARY) #pixels greater than thresh


    all_pixels = binary_image1

    binary_image1 = cv.medianBlur(binary_image1,5)
    binary_image2 = cv.medianBlur(binary_image2,5)
    for i in range(1):
        binary_image1 = cv.medianBlur(binary_image1,5)
        binary_image2 = cv.medianBlur(binary_image2,5)

    height, width, channels = img.shape

    mask_inv = cv.bitwise_not(binary_image2)
    merged_image = cv.bitwise_and(binary_image1, mask_inv)
    # merged_image = cv.bitwise_and(all_pixels,merged_image)

    contours, hierarchy = cv.findContours(merged_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contour_image = img.copy()

    for i, contour in enumerate(contours):
        if hierarchy[0][i][2] != -1:
            child_index = hierarchy[0][i][2]
            child_contour = contours[child_index]
        else:
            child_contour = contour        

        rect = cv.minAreaRect(child_contour)
        width, height = rect[1]
        width = round(width, 2)
        height = round(height, 2)

        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(contour_image,[box],0,(0,255,0),5)
        
        M = cv.moments(child_contour) #centroid
        try:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            wd = max(width,height)
            cv.putText(contour_image, str(wd),(cx,cy), cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color = (0, 0, 255), thickness=3)
        except:
            print('contour error')

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    text = str(len(contours))
    font = cv.FONT_HERSHEY_SIMPLEX
    position = (50, 50)
    font_scale = 2
    color = (0, 255, 0)  # BGR color (here, green)

    cv.putText(contour_image, text, position, font, font_scale, color, thickness=2)

    output_folder = 'RE_GEN'
    # output_folder = 'RE_GENERATED'
    processed_file_path = os.path.join(output_folder,file_name)
    cv.imwrite(processed_file_path,contour_image)
    # print("-----------------------------------------------------------------------")
    # cv.imwrite(processed_file_path,thresh)
