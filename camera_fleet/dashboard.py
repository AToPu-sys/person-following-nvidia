from __future__ import annotations
import math
import time

import cv2
import numpy as np

from .worker import CameraSnapshot


class FleetDashboard:
    def __init__(self, *, tile_width=640, tile_height=400, columns=2):
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.columns = columns
        self.window = "Camera Fleet | authorized feeds"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, tile_width * columns, tile_height * 2)

    def render(self, cameras: dict[str, CameraSnapshot]):
        ordered = sorted(cameras.items())
        if not ordered:
            return np.zeros((self.tile_height, self.tile_width, 3), np.uint8)
        rows = math.ceil(len(ordered) / self.columns)
        canvas = np.zeros((rows*self.tile_height, self.columns*self.tile_width, 3), dtype=np.uint8)
        for index, (camera_id, snap) in enumerate(ordered):
            col, row = index % self.columns, index // self.columns
            x, y = col*self.tile_width, row*self.tile_height
            tile = np.zeros((self.tile_height, self.tile_width, 3), dtype=np.uint8)
            if snap.frame is not None:
                video_height = self.tile_height - 58
                tile[:video_height] = cv2.resize(snap.frame, (self.tile_width, video_height))
            state_color = (55, 220, 90) if snap.status == "active" else (0, 180, 255)
            cv2.rectangle(tile, (0, 0), (self.tile_width, 35), (25, 30, 35), -1)
            cv2.putText(tile, camera_id, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .58, (245,245,245), 2, cv2.LINE_AA)
            state = snap.status.replace("_", " ").upper()
            cv2.putText(tile, state, (self.tile_width-190, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, state_color, 1, cv2.LINE_AA)
            cv2.rectangle(tile, (0, self.tile_height-58), (self.tile_width, self.tile_height), (25, 30, 35), -1)
            cv2.putText(tile, f"PEOPLE IN VIEW  {snap.people_visible}   |   {snap.fps:.1f} FPS",
                        (12, self.tile_height-22), cv2.FONT_HERSHEY_SIMPLEX, .48, (235,235,235), 1, cv2.LINE_AA)
            canvas[y:y+self.tile_height, x:x+self.tile_width] = tile
        cv2.imshow(self.window, canvas)
        return canvas

    def close(self):
        cv2.destroyWindow(self.window)

    def is_quit_pressed(self):
        return cv2.waitKey(1) & 0xFF in (ord("q"), 27)
