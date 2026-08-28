"""Reads frames from a video file, image sequence, or live drone feed, one frame at a time."""

import cv2


class VideoSource:
    def __init__(self, path_or_stream: str):
        self.path_or_stream = path_or_stream

    def frames(self):
        capture = cv2.VideoCapture(self.path_or_stream)
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video source: {self.path_or_stream}")
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                yield frame
        finally:
            capture.release()

    @property
    def fps(self) -> float:
        capture = cv2.VideoCapture(self.path_or_stream)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        capture.release()
        return fps

    @property
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of the source's frames."""
        capture = cv2.VideoCapture(self.path_or_stream)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return width, height
