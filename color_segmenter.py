#!/usr/bin/python3
import argparse
import copy
from functools import partial
from colorama import Fore, Back, Style, init
import cv2
import numpy as np
import json
import copy

init(autoreset=True)

# Function -----------------------------------
def onTrackbar(x):
    pass


def main():
    # Variables
    wname = "Original"
    wname_segmenter = 'Color Segmenter'
    ranges = {'h': {'min': 0, 'max': 256},
              's': {'min': 0, 'max': 256},
              'v': {'min': 0, 'max': 256}}
    limits = {}

    # configure opencv window
    cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(wname_segmenter, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('MinH', wname_segmenter, 0,   256, onTrackbar)
    cv2.createTrackbar('MaxH', wname_segmenter, 256, 256, onTrackbar)
    cv2.createTrackbar('MinS', wname_segmenter, 0,   256, onTrackbar)
    cv2.createTrackbar('MaxS', wname_segmenter, 256, 256, onTrackbar)
    cv2.createTrackbar('MinV', wname_segmenter, 0,   256, onTrackbar)
    cv2.createTrackbar('MaxV', wname_segmenter, 256, 256, onTrackbar)

    folder = './'
    image_bgr = cv2.imread(folder + 'frame0002.jpg')
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    
    while True:

        if image is None:
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            print(Fore.RED + Style.BRIGHT + 'Video is over, terminating.')
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            break  # video is over

        #image_b, image_g, image_r = cv2.split(image)

        min_b = cv2.getTrackbarPos('MinH', wname_segmenter)
        max_b = cv2.getTrackbarPos("MaxH", wname_segmenter)
        min_g = cv2.getTrackbarPos("MinS", wname_segmenter)
        max_g = cv2.getTrackbarPos("MaxS", wname_segmenter)
        min_r = cv2.getTrackbarPos("MinV", wname_segmenter)
        max_r = cv2.getTrackbarPos("MaxV", wname_segmenter)

        ranges["h"]["min"] = min_b
        ranges["h"]["max"] = max_b
        ranges["s"]["min"] = min_g
        ranges["s"]["max"] = max_g
        ranges["v"]["min"] = min_r
        ranges["v"]["max"] = max_r

        mins = np.array([ranges["h"]["min"], ranges["s"]["min"], ranges["v"]["min"]])
        maxs = np.array([ranges["h"]["max"], ranges["s"]["max"], ranges["v"]["max"]])

        mask = cv2.inRange(image, mins, maxs)
        #mask = mask.astype(np.bool)
        #mask = mask.astype(np.uint8)*255

        cv2.imshow(wname, image_bgr)
        cv2.imshow(wname_segmenter, mask)

        key = cv2.waitKey(20)

        if key == ord('q'):  # q for quit
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            print(Fore.WHITE + 'You pressed ' + Fore.CYAN + Style.BRIGHT + 'q' + Fore.WHITE + Style.RESET_ALL + '...' + Fore.RED + Style.BRIGHT + ' aborting' + Style.RESET_ALL)
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            print(Fore.YELLOW + Style.BRIGHT + '-- The program was terminated. --' + Style.RESET_ALL)
            break
        elif key == ord('w'):
            #buscar os min e max do r,g e b, que já estão disponiveis nos ranges
            limits['limits'] = copy.deepcopy(ranges)
            f = open("limits.json", "w")
            #transformar dicionario numa estrutura json e gravar estrutura num ficheiro
            f.write(json.dumps(limits, indent = 4))
            f.close()
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            print(Fore.GREEN + Style.BRIGHT + "Ranges saved successfully!")
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            print(Fore.WHITE + Style.BRIGHT + "-- Press " + Fore.CYAN + Style.BRIGHT + "q" + Fore.WHITE + Style.BRIGHT + " if you want to exit the program. --")


if __name__ == '__main__':
    main()