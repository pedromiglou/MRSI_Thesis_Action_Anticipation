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
    ranges = {'b': {'min': 0, 'max': 256},
              'g': {'min': 0, 'max': 256},
              'r': {'min': 0, 'max': 256}}
    limits = {}

    # configure opencv window
    cv2.namedWindow(wname, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(wname_segmenter, cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('MinB', wname_segmenter, 0,   256, onTrackbar)
    cv2.createTrackbar('MaxB', wname_segmenter, 256, 256, onTrackbar)
    cv2.createTrackbar('MinG', wname_segmenter, 0,   256, onTrackbar)
    cv2.createTrackbar('MaxG', wname_segmenter, 256, 256, onTrackbar)
    cv2.createTrackbar('MinR', wname_segmenter, 0,   256, onTrackbar)
    cv2.createTrackbar('MaxR', wname_segmenter, 256, 256, onTrackbar)

    folder = './'
    image = cv2.imread(folder + 'frame0002.jpg')
    
    while True:

        if image is None:
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            print(Fore.RED + Style.BRIGHT + 'Video is over, terminating.')
            print(Fore.WHITE + Style.BRIGHT + "-----------------------------" + Style.RESET_ALL)
            break  # video is over

        image_b, image_g, image_r = cv2.split(image)

        min_b = cv2.getTrackbarPos('MinB', wname_segmenter)
        max_b = cv2.getTrackbarPos("MaxB", wname_segmenter)
        min_g = cv2.getTrackbarPos("MinG", wname_segmenter)
        max_g = cv2.getTrackbarPos("MaxG", wname_segmenter)
        min_r = cv2.getTrackbarPos("MinR", wname_segmenter)
        max_r = cv2.getTrackbarPos("MaxR", wname_segmenter)

        ranges["b"]["min"] = min_b
        ranges["b"]["max"] = max_b
        ranges["g"]["min"] = min_g
        ranges["g"]["max"] = max_g
        ranges["r"]["min"] = min_r
        ranges["r"]["max"] = max_r

        mins = np.array([ranges['b']['min'], ranges["g"]["min"], ranges["r"]["min"]])
        maxs = np.array([ranges["b"]["max"], ranges["g"]["max"], ranges["r"]["max"]])

        mask = cv2.inRange(image, mins, maxs)
        #mask = mask.astype(np.bool)
        #mask = mask.astype(np.uint8)*255

        cv2.imshow(wname, image)
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