import cv2
import numpy as np

import argparse
import json

from os import path
import sys
from datetime import datetime

from color_segmenter import apply_mask


def get_centroid(mask):
    """
    function 'get_centroid': analyses the result of a mask being applied to an image (the result being
                            a binary image) in order to find its largest color segment, as well as its
                            centroid 
        INPUT:
            - mask: an image where a color segmentation mask has been applied; it is a BINARY
                    image where the white pixels represent that a given color has been detected
                    for that pixel in the original image, whereas black pixels signify the
                    absense of that same color
        OUTPUT:
            - (cX, cY): the X and Y coordinates (respectively) of the centroid of the biggest
                        object (white blob) in the 'mask' input
            - final_image: NOT a binary image; this is an RGB image with all the detected objects in the
                        'mask' input; the biggest object is colored green, whereas all the other ones
                        remain white; a red cross is placed in the position of the centroid
    """

    # find all contours (objects)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # if we detect objects, let's find the biggest one, make it green and calculate the centroid
    if cnts:

        # find the biggest object
        cnt = max(cnts, key=cv2.contourArea)

        # make it green (but still show other objects in white)
        biggest_obj = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(biggest_obj, [cnt], -1, 255, cv2.FILLED)
        biggest_obj = cv2.bitwise_and(mask, biggest_obj) # mask-like image with only the biggest object
        all_other_objs = cv2.bitwise_xor(mask, biggest_obj) # all other objects except the biggest one
        
        b = all_other_objs
        g = mask
        r = all_other_objs

        final_image = cv2.merge((b, g, r))

        # calculate centroid coordinates
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if (M["m00"]!=0) else None
        cY = int(M["m01"] / M["m00"]) if (M["m00"]!=0) else None

        # draw small red cross to indicate the centroid point
        if cX and cY:
            cv2.line(final_image, (cX-8, cY-8), (cX+8, cY+8), (0, 0, 255), 5)
            cv2.line(final_image, (cX+8, cY-8), (cX-8, cY+8), (0, 0, 255), 5)

    # if we don't detect any objects, we just show the mask as it is
    else:
        final_image = cv2.merge((mask, mask, mask))
        cX = None
        cY = None
    
    return (cX,cY), final_image


def draw(image, old_coords, coords, color, thickness):
    """
    function draw: draws on the canvas
        INPUT:
            - image:      this is our canvas, meaning it is the image we want to draw on
            - old_coords: last position of the pencil
            - coords:     new position of the pencil
            - color:      pencil color
            - thickness:  pencil thickness
        OUTPUT:
            - [return value]: there are three distinct cases for what the return value could
                            be, altough in all of them we return an RGB image
                            (1) if the new pencil coordinates are (None,None), then we do
                                not draw on the image, simply return it as it was given
                            (2) if the new pencil coordinates are DIFFERENT from (None, None)
                                but we do not have any previous coordinates (meaning that
                                old_coords is (None,None)), we simply draw a dot on the image
                                and return it
                            (3) if we have both the previous coordinates and the new ones, we
                                draw a line going from the previous to the new coordinates on
                                the image and return it
    """

    if (not coords[0]) or (not coords[1]):
        return image
    else:
        if (not old_coords[0]) or (not old_coords[1]):
            return cv2.circle(image, coords, thickness, color, -1)
        else:
            return cv2.line(image, (old_coords[0], old_coords[1]), (coords[0], coords[1]), color, thickness)


def main():
    """
    function main: initializes all the necessary elements of the program and performs the
                continuous color detection and drawing operations, as well as the key
                detection aspect of the program
    """

    # ------------ Initialization ------------

    # processing command line arguments
    parser = argparse.ArgumentParser(description='PSR AR Paint')
    parser.add_argument('-j', '--json', type=str, required=False, help='Use this argument to provide the path to the .json file with the color data.')
    args = vars(parser.parse_args())
    # if a path to a .json file is not provided, we assume it's the
    # limits.json file resulting from the execution of color_segmenter.py
    json_path = 'limits.json' if not args['json'] else args['json']

    # reading color information from .json file
    try:
        with open(json_path, 'r') as openfile:
            json_object = json.load(openfile)
            ranges = json_object['limits']
    # if the file doesn't exist, send out an error message and quit
    except FileNotFoundError:
        sys.exit('The .json file with the color data doesn\'t exist.')


    # setting up the video capture
    capture = cv2.VideoCapture(0)
    _, frame = capture.read() # initial frame just to figure out proper window dimensions

    # dimensions for all windows
    scale = 0.6
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)

    # continuing video capture setup
    camera_window = 'Camera capture'
    cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(camera_window, (window_width, window_height))

    # setting up a white canvas
    canvas = np.zeros([720,1280,3],dtype=np.uint8)
    canvas[:] = 255
    canvas_window = 'Canvas'
    cv2.namedWindow(canvas_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(canvas_window, (window_width, window_height))

    # setting up the window that shows the mask being applied
    mask_window = 'Masked capture'
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mask_window, (window_width, window_height))

    # define positions of each window on screen (this way, they don't overlap)
    cv2.moveWindow(camera_window, 200, 100)
    cv2.moveWindow(mask_window, 1000, 100)
    cv2.moveWindow(canvas_window, 600, 580)

    # ------------ Continuous Operation ------------

    # default pencil setup
    draw_color = (0,0,255) # starts off red
    draw_thickness = 5
    old_centroid_coords = (None,None)

    while True:

        # capture an image with the camera
        _, frame = capture.read()
        mask = apply_mask(frame, ranges)

        # show camera and canvas
        cv2.imshow(camera_window, frame)
        cv2.imshow(canvas_window, canvas)

        # calculate centroid of the largest color blob and show the mask being applied
        centroid_coords, detected_centroid = get_centroid(mask)
        cv2.imshow(mask_window, detected_centroid)

        # update the canvas and the pencil coordinates according to the most recent drawing movement
        canvas = draw(canvas, old_centroid_coords, centroid_coords, draw_color, draw_thickness)
        old_centroid_coords = centroid_coords


        # wait for a command
        pressedKey = cv2.waitKey(1) & 0xFF

        # 'q' key to quit the program
        if pressedKey == ord('q'):
            break

        # change pencil color
        elif pressedKey==ord('r'):
            draw_color = (0,0,255)
        elif pressedKey==ord('g'):
            draw_color = (0,255,0)
        elif pressedKey==ord('b'):
            draw_color = (255,0,0)

        # change pencil thickness
        elif pressedKey==ord('-'):
            if draw_thickness > 1:
                draw_thickness -= 4
        elif pressedKey==ord('+'):
            if draw_thickness < 40:
                draw_thickness += 4
        
        # clear canvas
        elif pressedKey==ord('c'):
            canvas = np.zeros([720,1280,3],dtype=np.uint8)
            canvas[:] = 255
            old_centroid_coords = (None,None)

        # save image
        elif pressedKey==ord('w'):
            today = datetime.now()
            formatted_date = today.strftime("%a_%b_%d_%H:%M:%S")
            image_name = 'drawing_' + formatted_date + '.png'
            cv2.imwrite(image_name, canvas)



if __name__ == '__main__':
    main()