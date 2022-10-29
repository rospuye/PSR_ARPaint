import cv2
# import numpy as np

import argparse
import json
# from functools import partial

from os import path
import sys
from datetime import datetime

from classes import *
from aux_functions import \
    get_centroid_position, \
    get_mouse_position, \
    new_draw_move, \
    redraw_on_frame, \
    apply_mask



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
        cv2.setMouseCallback(mask_window, mouse.update_mouse)

    # according to the pressing of the 's', 'e' or 'o' keys, this variable keeps up
    # with which mode we're in (which figure the user wants to draw);
    # if the user doesn't want to draw any figure but is simply in normal drawing
    # mode, this variable is None
    figure_mode = None

    # this variable keeps track of the current square or ellipse preview, should
    # the user be in the middle of drawing one;
    # otherwise, it's None
    figure_cache = None


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

        # update the history of draw moves
        # we only add the most recent move if:
        #   (1) we're on mouse mode AND the mouse is pressed
        #       OR
        #   (2) we're not on mouse mode
        if (use_mouse and mouse.pressed) or (not use_mouse):

            # free drawing mode
            if not figure_mode:
                draw_moves.append(new_draw_move(old_pencil_coords, pencil_coords, draw_color, draw_thickness, usp))

            # figure mode and we're detecting the pencil
            elif pencil_coords!=(None,None):

                # if we already have a figure in cache, its origin remains the same;
                # if not, its origin is set as the current pencil coordinates
                # note: the origin of a figure is it's top-left corner for a rectangle/square or
                # an ellipse, and its center for a circle
                origin = figure_cache.origin if figure_cache else pencil_coords

                # update the figure cache with the figure's new positioning
                if figure_mode=='square':
                    figure_cache = Square(origin, pencil_coords, draw_color, draw_thickness)
                elif figure_mode=='ellipse':
                    figure_cache = Ellipse(origin, pencil_coords, draw_color, draw_thickness)
                elif figure_mode=='circle':
                    figure_cache = Circle(origin, pencil_coords, draw_color, draw_thickness)

                # update draw_moves with the new figure in cache
                draw_moves[-1] = figure_cache

            # figure mode but we can't detect the pencil
            elif pencil_coords==(None,None):
                # if there's any figure in the cache, we make it grey;
                # this signals to the user that the current figure is impossible to edit at the moment, but that
                # it will continue the positioning process once the pencil coordinates are detected once more;
                # if the user purposefully chooses to hide the pencil pointer and deactivate figure mode while
                # in this state, that allows them to give up on drawing this figure
                if figure_cache:
                    figure_cache.color = (190,190,190)
                    draw_moves[-1] = figure_cache

        # redraw history of moves on new frame
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

        # draw figure
        elif pressedKey==ord('s') or pressedKey==ord('e') or pressedKey==ord('o'):

            # determine which figure it is
            figure = 'square' if pressedKey==ord('s') else ('ellipse' if pressedKey==ord('e') else 'circle')

            # if we were previously drawing another type of figure, we clean the figure cache and remove
            # the previous figure from draw_moves; this code will also execute if we were previously
            # free drawing, but it doesn't make a difference
            if figure_mode!=figure:
                figure_cache = None
                draw_moves = draw_moves[:-1]

            # if we were already in the correct figure's drawing mode for the pressed key, we set figure_mode
            # to None because it means we're pressing the figure's key for a second time (a.k.a. the user is
            # finished positioning it); otherwise, we set the mode to the inteded figure for the detected key
            # press
            figure_mode = figure if (figure_mode!=figure) else None

            # if we just finished positioning a figure, we clean the cache
            if figure_mode != figure:

                # if we have a grey figure and the figure mode is deactivated, it means the user gave up on
                # drawing that figure, and thus we also remove it from the draw_moves list
                if figure_cache and figure_cache.color == (190,190,190):
                    draw_moves = draw_moves[:-1]

                figure_cache = None



if __name__ == '__main__':
    main()