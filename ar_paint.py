import cv2
import numpy as np

import argparse
import json
from functools import partial

from os import path
import sys
from datetime import datetime

from color_segmenter import apply_mask


class Mouse:
    """
    class 'Mouse': this class is only ever instantiated if we run the program with the -m flag, meaning
                we wish to use the mouse pointer instead of the color centroid as a pencil; in that case,
                the instantiation of Mouse keeps track of the coordinates of the mouse pointer, thus
                avoiding the use of global variables
    """

    def __init__(self):
        self.coords = (None,None)

    def update_coords(self,event,x,y,flags,param):
        self.coords = (x,y)

class Dot:
    """
    class 'Dot': represents a drawn dot on the canvas; this class allows us to instantiate dots drawn on
                the canvas (camera frame) so that they might be redrawn on each new frame
    """

    def __init__(self, coords, thickness, color):
        self.coords = coords
        self.thickness = thickness
        self.color = color

class Line:
    """
    class 'Line': represents a drawn line on the canvas; this class allows us to instantiate lines drawn on
                the canvas (camera frame) so that they might be redrawn on each new frame
    """

    def __init__(self, old_coords, coords, thickness, color):
        self.old_coords = old_coords
        self.coords = coords
        self.thickness = thickness
        self.color = color

def get_centroid_position(mask):
    """
    function 'get_centroid_position': analyses the result of a mask being applied to an image (the result being
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
        if cX: # it's enough to check either cX or cY, if one is None then both are None
            cv2.line(final_image, (cX-8, cY-8), (cX+8, cY+8), (0, 0, 255), 5)
            cv2.line(final_image, (cX+8, cY-8), (cX-8, cY+8), (0, 0, 255), 5)

    # if we don't detect any objects, we just show the mask as it is
    else:
        final_image = cv2.merge((mask, mask, mask))
        cX = None
        cY = None
        
    return (cX,cY), final_image

def get_mouse_position(mouse):
    """
    function 'get_mouse_position': if we are running the script with the -m flag, it returns the mouse
                                coordinates on a black screen
        INPUT:
            - mouse: the instantiation of the Mouse class, which keeps the mouse pointer coordinate
                    information
        OUTPUT:
            - (cX, cY): the X and Y coordinates (respectively) of the mouse pointer
            - final_image: a black screen where a red cross is placed in the position of the mouse
                        coordinates
    """

    final_image = np.zeros([720,1280,3],dtype=np.uint8) # black screen
    cX = mouse.coords[0]
    cY = mouse.coords[1]
    # drawing the red cross
    if cX:
        cv2.line(final_image, (cX-8, cY-8), (cX+8, cY+8), (0, 0, 255), 5)
        cv2.line(final_image, (cX+8, cY-8), (cX-8, cY+8), (0, 0, 255), 5)
    return (cX,cY), final_image


def new_draw_move(old_coords, coords, color, thickness, usp):
    """
    function new_draw_move: returns the class instantiation of a new drawing move performed by
                            the end-user
        INPUT:
            - old_coords: last position of the pencil
            - coords:     new position of the pencil
            - color:      pencil color
            - thickness:  pencil thickness
            - usp:        boolean that indicates whether or not we're performing shake prevention
        OUTPUT:
            - [return value]: there are three distinct cases for what the return value could
                            be
                            (1) if the new pencil coordinates are (None,None), then we do
                                not draw on the image, and thus return None
                            (2) if the new pencil coordinates are DIFFERENT from (None, None)
                                but we do not have any previous coordinates (meaning that
                                old_coords is (None,None)), we simply return a Dot object; this
                                scenario also happens if the program is ran with the shake detection
                                option activated and the difference between old_coords and coords
                                surpasses a certain threshold
                            (3) if we have both the previous coordinates and the new ones, we
                                return a Line object going from the previous to the new coordinates
    """

    if coords!=(None,None):
        # difference along the x and y axes between the pencil's last and current position;
        # if either of these differences is too big and we ran this program with the -usp flag,
        # shake detection will be activated
        diffX = abs(old_coords[0] - coords[0]) if old_coords[0] else None
        diffY = abs(old_coords[1] - coords[1]) if old_coords[1] else None

        # TODO: potentially need to adjust the threshold of the shake detection
        if old_coords==(None,None) or \
            (usp and (diffX>50 or diffY>50)): # this line performs shake detection
            return Dot(coords,thickness,color)
        else:
            return Line(old_coords,coords,thickness,color)
    return None

def redraw_on_frame(image, draw_moves):
    """
    function redraw_on_frame: re-draws all the user's move history on the newly capture camera frame
        INPUT:
            - image:      canvas on which to draw (that being the new camera frame)
            - draw_moves: history of drawing moves performed by the user
        OUTPUT:
            - image: altered canvas, already with the drawings on it
    """

    for move in draw_moves:
        # draw on image
        if type(move) is Dot:
            cv2.circle(image, move.coords, move.thickness, move.color, -1)
        elif type(move) is Line:
            cv2.line(image, (move.old_coords[0], move.old_coords[1]), (move.coords[0], move.coords[1]), move.color, move.thickness)
    return image

def main():
    """
    function main: initializes all the necessary elements of the program and performs the
                continuous color detection and drawing operations, as well as the key
                detection aspect of the program
    """

    # ------------ Initialization ------------

    # processing command line arguments
    parser = argparse.ArgumentParser(description='PSR AR Paint')
    parser.add_argument('-j', '--json', type=str, required=False, help='provide the path to the .json file with the color data')
    parser.add_argument('-usp', '--use_shake_prevention', action='store_true', help='use shake prevention while drawing')
    parser.add_argument('-m', '--mouse', action='store_true', help='test the program with the mouse pointer instead of the color centroid')
    args = vars(parser.parse_args())

    # if a path to a .json file is not provided, we assume it's the
    # limits.json file resulting from the execution of color_segmenter.py
    json_path = 'limits.json' if not args['json'] else args['json']

    # boolean that determines if shake prevention is to be used or not
    usp = args['use_shake_prevention']
    # boolean that determines if the mouse pointer is to be used or not
    use_mouse = args['mouse']

    # reading color information from .json file
    try:
        with open(json_path, 'r') as openfile:
            json_object = json.load(openfile)
            ranges = json_object['limits']
    # if the file doesn't exist, send out an error message and quit
    except FileNotFoundError:
        sys.exit('The .json file with the color data doesn\'t exist.')

    # list of all the draw moves done so far
    draw_moves = []

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

    # setting up the window that shows the mask being applied
    mask_window = 'Masked capture'
    cv2.namedWindow(mask_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(mask_window, (window_width, window_height))

    # define positions of each window on screen (this way, they don't overlap)
    cv2.moveWindow(camera_window, 200, 100)
    cv2.moveWindow(mask_window, 1000, 100)

    # if we're going to use mouse coordinates in place of the centroid, we need to
    # keep track of the mouse
    if use_mouse:
        mouse = Mouse()
        cv2.setMouseCallback(mask_window, mouse.update_coords)

    # ------------ Continuous Operation ------------

    # default pencil setup
    draw_color = (0,0,255) # starts off red
    draw_thickness = 5
    old_pencil_coords = (None,None)

    while True:

        # capture an image with the camera
        _, frame = capture.read()
        mask = apply_mask(frame, ranges)


        # calculate centroid of the largest color blob and show the mask being applied
        pencil_coords, detected_pencil = get_centroid_position(mask) if not use_mouse else get_mouse_position(mouse)
        cv2.imshow(mask_window, detected_pencil)

        # update the frame and the pencil coordinates according to the most recent drawing movement
        draw_moves.append(new_draw_move(old_pencil_coords, pencil_coords, draw_color, draw_thickness, usp))
        frame = redraw_on_frame(frame,draw_moves)
        old_pencil_coords = pencil_coords

        # show frame
        cv2.imshow(camera_window, frame)

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
            draw_moves = []
            old_pencil_coords = (None,None)

        # save image
        elif pressedKey==ord('w'):
            today = datetime.now()
            formatted_date = today.strftime("%a_%b_%d_%H:%M:%S")
            image_name = 'drawing_' + formatted_date + '.png'
            cv2.imwrite(image_name, frame)



if __name__ == '__main__':
    main()