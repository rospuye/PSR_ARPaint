import cv2
import numpy as np
from functools import partial
import json
from os import path

def update_range_dict(val,ranges,color,bound):
    ranges[color][bound] = val

# Alter image according to the RBG parameters
def alter_image(ranges, image, window_name):

    lows = (ranges['B']['min'], ranges['G']['min'], ranges['R']['min'])
    highs = (ranges['B']['max'], ranges['G']['max'], ranges['R']['max'])

    final_image = cv2.inRange(image, lows, highs)
    cv2.imshow(window_name, final_image)


def main():

    write = False

    # default values to start with
    fileAlreadyExists = path.exists('limits.json')
    if fileAlreadyExists:
        with open('limits.json', 'r') as openfile:
            json_object = json.load(openfile)
            ranges = json_object['limits']
    else:
        ranges = { 'B':{'max': 255, 'min': 0}, 'G':{'max': 255, 'min': 0}, 'R':{'max': 255, 'min': 0} }

    capture = cv2.VideoCapture(0)
    window_name = 'Color segmentation'
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
    slider_max = 255

    # Trackbars for R, G and B dimensions

    cv2.createTrackbar('R min', window_name , 0, slider_max, partial(update_range_dict, ranges=ranges, color='R', bound='min'))
    cv2.createTrackbar('R max', window_name , 0, slider_max, partial(update_range_dict, ranges=ranges, color='R', bound='max'))
    
    cv2.createTrackbar('G min', window_name , 0, slider_max, partial(update_range_dict, ranges=ranges, color='G', bound='min'))
    cv2.createTrackbar('G max', window_name , 0, slider_max, partial(update_range_dict, ranges=ranges, color='G', bound='max'))
    
    cv2.createTrackbar('B min', window_name , 0, slider_max, partial(update_range_dict, ranges=ranges, color='B', bound='min'))
    cv2.createTrackbar('B max', window_name , 0, slider_max, partial(update_range_dict, ranges=ranges, color='B', bound='max'))

    while True:

        # Capture frame-by-frame and display
        _, frame = capture.read()
        alter_image(ranges,frame,window_name)

        # Quit
        if cv2.waitKey(33) == ord('q'):
            break
        # Write to file
        elif cv2.waitKey(33) == ord('w'):
            write = True
            break

    if write:
        # write to file
        data = {'limits' : ranges}
        json_object = json.dumps(data, indent=4)
        with open("limits.json", "w") as outfile:
            outfile.write(json_object)



if __name__ == '__main__':
    main()