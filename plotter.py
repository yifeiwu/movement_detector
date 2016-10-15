#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import sys

sys.path.append('/home/david/healthhack/bin/lib') #where your lib/cs2.so file is located


'''
Plotter
====================

Usage
-----
plotter.py [<source.csv>]
output: png plot file
'''

import numpy as np
import pdb
from numpy import linalg as LA
import cv2
import video
from common import anorm2, draw_str
import time 
import csv
import logging
import ntpath
import matplotlib.pyplot as plt

def main():

    x=[]
    y=[]
    with open(sys.argv[1], 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            x.append(float(row[1])+float(row[2])/60)
            y.append(row[3])


    plt.plot(x, y, 'ro')
    plt.savefig(ntpath.basename(sys.argv[1])+'-output.png')


if __name__ == '__main__':
  main()



