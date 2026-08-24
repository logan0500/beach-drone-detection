"""Reads frames from a video file or a live drone feed, one frame at a time."""


class VideoSource:
    def __init__(self, path_or_stream: str):
        self.path_or_stream = path_or_stream

    def frames(self):
        raise NotImplementedError
