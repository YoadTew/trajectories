#!/usr/bin/env python
'''
Lucas-Kanade tracker
====================
Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.
Usage
-----
lk_track.py [<video_source>]
Keys
----
ESC - exit
'''
import numpy as np
import cv2

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
class App:
    def __init__(self, video_src):
        self.track_len = 10000
        self.detect_interval = 5
        self.tracks = []
        self.cam =cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.history_tracks = []
    def run(self):
        while self.frame_idx < 600:
            ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
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
                        self.history_tracks.append((self.tracks, self.frame_idx - 1))
                        continue
                    # tr.append((x, y,self.frame_idx))
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    #self.history_tracks.append(tr)
                    #cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                # cv2.polylines(vis, [np.int32(tr) for tr in self.tracks], False, (0, 255, 0))
                #draw_str(vis, (20, 20), 'track count: %d' % len(self.tracks))
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                #for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    #cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        # self.tracks.append([x, y, self.frame_idx])
                        self.tracks.append([(x, y)])
            self.frame_idx += 1
            self.prev_gray = frame_gray
            #cv2.imwrite('/home/work/Lktrack/lk_track'+ str(self.frame_idx)+'.jpg', vis)
        # cv2.polylines(frame, [np.int32(tr)[:,:2] for tra in self.history_tracks for tr in tra], False, (0, 255, 0))
        draw_history_tracks = [x[0] for x in self.history_tracks]
        cv2.polylines(frame, [np.int32(tr) for tra in draw_history_tracks for tr in tra], False, (0, 255, 0))
        cv2.imwrite('/home/work/Lktrack/lk_track_enddd.jpg', frame)

def main():
    video_src = r'/tmp/PEDESTRIANS.avi'
    App(video_src).run()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()