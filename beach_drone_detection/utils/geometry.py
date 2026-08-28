"""Small geometry helpers shared across detectors."""

import math


def box_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def point_in_polygon(point, polygon) -> bool:
    """polygon is a list of (x, y) vertices. Uses OpenCV's point-in-polygon test."""
    import numpy as np
    import cv2

    poly = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(poly, point, False) >= 0
