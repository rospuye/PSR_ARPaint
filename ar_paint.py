from asyncore import read
from fileinput import filename
import cv2
from datetime import datetime
import argparse
from readchar import readkey, key
def main():
    parser = argparse.ArgumentParser(
        description='Typer test. The program tests the user for typing accuracy on the keyboard.')
    parser.add_argument('-h','--help', action='store_true',
        help='show this help message and exit.')
    parser.add_argument('-j limits.json','--json JSON', type=int, required=True, 
        help='Full path to json file.')
    args = vars(parser.parse_args())
    limits=args['--json JSON']
    while True:
        capture=cv2.VideoCapture(0)
        cv2.namedWindow(capture,cv2.WINDOW_AUTOSIZE)
        _, image =capture.read()
        dimensions = capture.shape
        tela=capture.fill(255)
        tela.read()
        mask = cv2.inRange(image, limits)
        mask.read()
        ret, thresh = cv2.threshold(mask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # You need to choose 4 or 8 for connectivity type
        connectivity = 4  
        # Perform the operation
        output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix
        labels = output[1]
        # The third cell is the stat matrix
        stats = output[2]
        # The fourth cell is the centroid matrix
        centroids = output[3]
            
        cv2.line(capture,centroids,(0,0,255))

        color=(255,255,255)
        thickness=1
        cv2.line(tela,centroids,color,thickness)
        if readkey() =='r':
            color=(0,0,255)
        elif readkey()=='g':
            color=(0,255,0)
        elif readkey()=='b':
            color=(255,0,0)
        if readkey() =='+':
            thickness +=1
        elif readkey() =='-':
            thickness -=1
        if readkey()=='c':
            tela=capture.fill(255)
        if readkey=='w':
            filename=str('drawing_'+str(datetime.now()))
            cv2.imwrite(filename, tela)
        if cv2.waitkey(33)==ord('q'):
            break
if __name__ == '__main__':
    main()