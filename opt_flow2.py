#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function
import sys

sys.path.append('/home/david/healthhack/bin/lib') #where your lib/cs2.so file is located


'''
Movement Detector
====================

Based on the Lucas-Kanade optical flow demo code in OpenCV.
Usage
-----
opt_flow2.py [<video_source>]
output: video with flowlines or csv with timepoints where significant movement is detected
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
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 700,
                       qualityLevel = 0.1,
                       minDistance = 7,
                       blockSize = 7 )

class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src)
        self.video_src = video_src
        self.frame_idx = 0

    def run(self):

        dir(self.cam)
        #output will not work if these don't match source size
        frame_width =  720
        frame_height=  576

        #fgbg = cv2.createBackgroundSubtractorMOG2() #enable background thresholding

        #kernel = np.ones((5,5),np.float32)/25 #gaussian smoothing

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (frame_width,frame_height))

        ofile  = open('ttest.csv', "wb")
        writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        start_time = time.time()


        while True:
            ret, frame = self.cam.read()
            #frame = cv2.filter2D(frame,-1,kernel) #gaussian smoothing
            #fgmask = fgbg.apply(frame) #bg thresholding
            #frame_gray = fgmask #bgthresholding

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            maxdisp = [0]
            significance = False

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (x, y), 2, (0, 0, 255), -1) #draw points
                self.tracks = new_tracks

                cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 0, 255)) #draw vectors
                
                for tr in self.tracks:
                    diff = np.array(tr[1]) - np.array(tr[0])
                    maxdisp.append(LA.norm(diff,2))
                #frame statistics

                draw_str(vis, (20, 20), 'Source: %s @ %d:%d' % (ntpath.basename(self.video_src), int(self.frame_idx/(60*25)), (self.frame_idx/25)%60 ))
                draw_str(vis, (20, 40), 'Flow Magnitude: %d' % max(maxdisp)) 
                
                #output
                significance = max(maxdisp)> 3
                if significance:
                    row = [self.frame_idx, int(self.frame_idx/(60*25)), (self.frame_idx/25)%60, max(maxdisp)]
                    writer.writerow(row)
                    #elapsed_time = time.time() - start_time
                    logging.warning( "Processing video at %d:%d. Displacement Magnitude %r." % (int(self.frame_idx/(60*25)), (self.frame_idx/25)%60, max(maxdisp)  ))


            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray

            #cv2.imshow('lk_track', vis)
            out.write(vis)


            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                out.release()
                break

def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()
    ofile.close()

if __name__ == '__main__':
    main()
