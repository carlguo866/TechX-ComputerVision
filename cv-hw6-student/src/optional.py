import cv2
import numpy as np
from functools import reduce


def preprocess_crop_black(image):
    _, mask = cv2.threshold(image, 1.0, 255.0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = reduce(lambda x, y: x if len(x) >= len(y) else y, contours)
    hull_points = cv2.convexHull(largest_contour, clockwise=True)
    max = 0;
    max_points = []
    for hull_point_i in hull_points:
        hull_point_i = hull_point_i[0]
        for hull_point_j in hull_points:
            hull_point_j = hull_point_j[0]
            #print(abs(hull_point_j[1] - hull_point_i[1]))
            area = abs(hull_point_j[0] - hull_point_i[0]) * abs(hull_point_i[1] - hull_point_j[1])
            if area > max:
                max = area;
                max_points = (hull_point_i,hull_point_j)
    print(max_points)
    return max_points
