import cv2
from functools import partial
import json
from os import path

from aux_functions import update_range_dict, apply_mask


def main():
    """
    function main: initializes all the necessary elements of the program and performs the continuous color
                segmentation operation, as well as the key detection aspect of the program
    """

    # get the color range values to start with
    # if we already have a 'limits.json' file in the directory, we get the previously saved values from there
    fileAlreadyExists = path.exists('limits.json')
    if fileAlreadyExists:
        with open('limits.json', 'r') as openfile:
            json_object = json.load(openfile)
            ranges = json_object['limits']
    # if not, we start with default values
    else:
        ranges = { 'B':{'max': 255, 'min': 0}, 'G':{'max': 255, 'min': 0}, 'R':{'max': 255, 'min': 0} }

    # set up video capture
    capture = cv2.VideoCapture(0)
    _, frame = capture.read()

    # figure out window size from the size of the frames captured by the camera
    scale = 0.55
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0])

    # finish up the video capture setup
    window_name = 'Color segmentation'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, window_width, window_height)

    # max value for all the trackbars
    slider_max = 255

    # trackbars for R, G and B dimensions

    cv2.createTrackbar('R min', window_name , ranges['R']['min'], slider_max, partial(update_range_dict, ranges=ranges, color='R', bound='min'))
    cv2.createTrackbar('R max', window_name , ranges['R']['max'], slider_max, partial(update_range_dict, ranges=ranges, color='R', bound='max'))
    
    cv2.createTrackbar('G min', window_name , ranges['G']['min'], slider_max, partial(update_range_dict, ranges=ranges, color='G', bound='min'))
    cv2.createTrackbar('G max', window_name , ranges['G']['max'], slider_max, partial(update_range_dict, ranges=ranges, color='G', bound='max'))
    
    cv2.createTrackbar('B min', window_name , ranges['B']['min'], slider_max, partial(update_range_dict, ranges=ranges, color='B', bound='min'))
    cv2.createTrackbar('B max', window_name , ranges['B']['max'], slider_max, partial(update_range_dict, ranges=ranges, color='B', bound='max'))

    # color segmentation continuous operation
    while True:

        # Capture frame-by-frame and display
        _, frame = capture.read()

        # apply color segmentation mask to the recently captured frame 
        mask = apply_mask(frame,ranges)
        cv2.imshow(window_name, mask)

        # wait for a command
        pressedKey = cv2.waitKey(1) & 0xFF

        # Quit
        if pressedKey == ord('q'):
            break
        # Write to file
        elif pressedKey == ord('w'):
            data = {'limits' : ranges}
            json_object = json.dumps(data, indent=4)
            with open("limits.json", "w") as outfile:
                outfile.write(json_object)
            break



if __name__ == '__main__':
    main()