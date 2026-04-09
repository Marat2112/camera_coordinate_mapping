import numpy as np
import cv2


def compute_homography(src_points, dst_points):
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)

    H, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )

    return H, mask


def apply_homography(H, point):
    x, y = point

    p = np.array([x, y, 1.0])
    p_transformed = H @ p

    if p_transformed[2] == 0:
        return 0.0, 0.0

    p_transformed = p_transformed / p_transformed[2]

    return float(p_transformed[0]), float(p_transformed[1])