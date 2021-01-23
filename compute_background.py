#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import sys

import cv2

from utils.motion_detection import Background


def main():
    args = argparse.ArgumentParser()
    args.add_argument("-v", "--videos", nargs='+', required=True,
                      help="path to the dir with videos")
    args.add_argument("-o", "--output", required=True, help="output dir")
    args.add_argument("--no-show", action='store_true', help="don't show window")

    config = args.parse_args()

    background = Background(
        no_average=False,
        skip=10 if len(config.videos) == 1 else len(config.videos)*5,
        take=10 if len(config.videos) == 1 else len(config.videos),
        use_last=30,
    )

    cap = [cv2.VideoCapture(video) for video in config.videos]
    frame_id = 0
    while True:
        
        decoded = [cap[i].read() for i in range(len(cap))]
        rets = [d[0] for d in decoded]
        frames = [d[1] for i,d in enumerate(decoded) if rets[i]]
        if len(frames) == 0:
            break

        for frame in frames:
            background.update(frame)

        if not config.no_show:
            bg = cv2.resize(background.background_color.copy(), (1280, 768))
            cv2.putText(bg, f'frame: {frame_id}', (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.imshow('Current Background', bg)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                sys.exit()
        frame_id += 1

    cv2.destroyAllWindows()
    for c in cap:
        c.release()
    
    background_color = background.background_color.copy()
    background_bw = background.background.copy()

    for video in config.videos:
        cv2.imwrite('{}/{}.bmp'.format(config.output, Path(video).stem), background_color)
        cv2.imwrite('{}/{}_bw.bmp'.format(config.output, Path(video).stem), background_bw)


if __name__ == "__main__":
    main()
