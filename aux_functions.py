import cv2
import numpy as np
from classes import Circle, Dot, Ellipse, Line, Square

# This file contains auxiliary functions used in the color_segmenter.py and ar_paint.py scripts.


# -----------------------------------------------------
#                   COLOR SEGMENTER
# -----------------------------------------------------

def update_range_dict(val,ranges,color,bound):
    """
    function update_range_dict: updates the dictionary that holds the valid ranges of each
                                of the 3 color channels for color segmentation with new
                                trackbar values
        INPUT:
            - val: new trackbar value, resulting from end-user's manipulation of the
                interface's trackbars
            - ranges: the dictionary holding the valid RBG ranges
            - color: the color channel that needs to be altered (so, either 'R', 'G' or 'B')
            - bound: the bound that needs to be altered (so, either 'min' or 'max')
    """
    ranges[color][bound] = val


# -----------------------------------------------------
#                       AR PAINT
# -----------------------------------------------------

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
        elif type(move) is Square:
            cv2.rectangle(image, move.origin, move.end_point, move.color, move.thickness)
        elif type(move) is Ellipse:
            cv2.ellipse(image, move.center, move.axes, move.angle, move.startAngle, move.endAngle, move.color, move.thickness)
        elif type(move) is Circle:
            cv2.circle(image, move.origin, move.radius, move.color, move.thickness)
    return image


# -----------------------------------------------------
#                         BOTH
# -----------------------------------------------------

def apply_mask(image, ranges):
    """
    function apply_mask: applies a color segmentation mask to an image
        INPUT:
            - image: original image we want to apply the mask to
            - ranges: this is basically the mask itself, except instead of being a binary image with which
                    we would perform a logical operation to the original image, it is a dictionary
                    indicating the valid ranges of values for each of the color channels - R, G and B
        OUTPUT:
            - [return value]: a binary image where the white pixels represent pixels in the original image
                            that stood within the valid ranges for all three color channels, whereas black
                            pixels represent pixels in the original image where at least one of the three
                            ranges was violated; this tells us which pixels in the original image were within
                            the valid color range we wish to detect
    """

    lows = (ranges['B']['min'], ranges['G']['min'], ranges['R']['min'])
    highs = (ranges['B']['max'], ranges['G']['max'], ranges['R']['max'])
    return cv2.inRange(image, lows, highs)
