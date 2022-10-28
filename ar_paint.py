import cv2
import argparse
from os import path
import json
import numpy as np
from datetime import datetime


# TODO: 
# - figure out window resizing stuff (low priority)

def apply_mask(image, ranges):
    lows = (ranges['B']['min'], ranges['G']['min'], ranges['R']['min'])
    highs = (ranges['B']['max'], ranges['G']['max'], ranges['R']['max'])
    return cv2.inRange(image, lows, highs)

def get_centroid(mask):

    # find all contours (objects)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if cnts:
        cnt = max(cnts, key=cv2.contourArea) # biggest object

        biggest_obj = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(biggest_obj, [cnt], -1, 255, cv2.FILLED)
        biggest_obj = cv2.bitwise_and(mask, biggest_obj) # mask-like image with only the biggest object
        all_other_objs = cv2.bitwise_xor(mask, biggest_obj)
        
        b = all_other_objs
        g = mask
        r = all_other_objs

        final_image = cv2.merge((b, g, r))

        # calculate centroid
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"]) if (M["m00"]!=0) else None
        cY = int(M["m01"] / M["m00"]) if (M["m00"]!=0) else None

        # draw the centroid
        if cX and cY:
            cv2.line(final_image, (cX-8, cY-8), (cX+8, cY+8), (0, 0, 255), 5)
            cv2.line(final_image, (cX+8, cY-8), (cX-8, cY+8), (0, 0, 255), 5)

    # if no detected objects, show black canvas
    else:
        # TODO: yuck
        final_image = cv2.merge((mask, mask, mask))
        cX = None
        cY = None
    
    return (cX,cY), final_image

def draw(image, old_coords, coords, color, thickness):
    if (not coords[0]) or (not coords[1]):
        return image
    else:
        if (not old_coords[0]) or (not old_coords[1]):
            return cv2.circle(image, coords, thickness, color, -1)
        else:
            return cv2.line(image, (old_coords[0], old_coords[1]), (coords[0], coords[1]), color, thickness)


def main():
    
    # ------------ Initialization ------------

    # processing command line arguments
    parser = argparse.ArgumentParser(description='PSR AR Paint')
    parser.add_argument('-j', '--json', action='store_true', help='Use this argument to provide the path to the .json file with the color data.')
    args = vars(parser.parse_args())
    json_path = 'limits.json' if not args['json'] else args['json']

    # reading color information from .json file
    try:
        with open(json_path, 'r') as openfile:
            json_object = json.load(openfile)
            ranges = json_object['limits']
    except:
        raise ValueError('The .json file with the color data doesn\'t exist.')

    # setting up the video capture
    capture = cv2.VideoCapture(0)
    camera_window = 'Camera capture'
    cv2.namedWindow(camera_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(camera_window, (600, 800))

    # setting up a white canvas
    canvas = np.zeros([720,1280,3],dtype=np.uint8)
    canvas[:] = 255
    canvas_window = 'Canvas'
    cv2.namedWindow(canvas_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(canvas_window, 600, 800)

    # setting up the window that shows the mask being applied
    mask_window = 'Masked capture'
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mask_window, 600, 800)

    # ------------ Continuous Operation ------------

    draw_color = (0,0,255) # default is red
    draw_thickness = 5
    old_centroid_coords = (None,None)

    while True:

        # capture an image with the camera
        _, frame = capture.read()
        mask = apply_mask(frame, ranges) # 2-dimensional

        cv2.imshow(camera_window, frame)
        cv2.imshow(canvas_window, canvas)

        centroid_coords, detected_centroid = get_centroid(mask)
        cv2.imshow(mask_window, detected_centroid)

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