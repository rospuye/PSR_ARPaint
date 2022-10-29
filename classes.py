import cv2
from math import sqrt

# This file contains all classes used in the ar_paint.py script.

class Mouse:
    """
    class 'Mouse': this class is only ever instantiated if we run the program with the -m flag, meaning
                we wish to use the mouse pointer instead of the color centroid as a pencil; in that case,
                the instantiation of Mouse keeps track of the coordinates of the mouse pointer, as well as
                if we're pressing the left mouse button or not, thus avoiding the use of global variables
    """

    def __init__(self):
        self.coords = (None,None)
        self.pressed = False

    def update_mouse(self,event,x,y,flags,param):
        self.coords = (x,y)

        if event == cv2.EVENT_LBUTTONDOWN:
            self.pressed = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.pressed = False


# Free drawing moves

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


# Geometrical figures

class Figure:
    """
    class 'Figure': a geometric figure with a given origin, pencil color and pencil thickness
    """

    def __init__(self, origin, color, thickness):
        self.origin = origin
        self.color = color
        self.thickness = thickness

class Square(Figure):
    """
    class 'Square': a square or rectangle; it inherits from Figure
                    - its origin is its top-left vertix
                    - the pencil is is the current pencil location on the canvas and it translated
                    to the position of the bottom-right vertix
    """

    def __init__(self, origin, pencil, color, thickness):
        Figure.__init__(self, origin, color, thickness)
        self.end_point = pencil

class Ellipse(Figure):
    """
    class 'Ellipse': an ellipse; it inherits from Figure
                    - its origin is its top-left point
                    - the pencil is the current pencil location on the canvas
                    - the center and axes length of the ellipse are calculated according to the
                    coordinates of both its origin and the pencil position
                    - the start and end angles of the ellipse, as well as its rotation angle, are
                    all constant values
    """
    
    def __init__(self, origin, pencil, color, thickness):
        meanX = (pencil[0]-origin[0])/2
        meanY = (pencil[1]-origin[1])/2

        Figure.__init__(self, origin, color, thickness)
        self.center = (round(meanX + origin[0]), round(meanY + origin[1]))
        self.axes = (round(abs(meanX)), round(abs(meanY)))
        self.angle = 0
        self.startAngle = 0
        self.endAngle = 360

class Circle(Figure):
    """
    class 'Circle': a circle; it inherits from Figure
                    - its origin is its center
                    - the edge is the current pencil location on the canvas and it translates to a point
                    in the circumference used in combination with its center to calculate the circle's
                    radius
    """

    def __init__(self, center, edge, color, thickness):
        diffX = edge[0] - center[0]
        diffY = edge[1] - center[1]

        Figure.__init__(self, center, color, thickness)
        self.radius = round(sqrt( diffX**2 + diffY**2 ))

