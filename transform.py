import cv2
import numpy as np
from face_detect import get_points
import math
from collections import namedtuple


Coord = namedtuple('Coord', 'x y')
RESIZE_H_W = (1080, 1980)


class Template:
    def __init__(self, img_path, frontal_face_detector, face_landmark_detector):
        self.points = get_points(cv2.imread(img_path), frontal_face_detector, face_landmark_detector)
        self.l_eye = _extract_l_eye_corners(self.points)
        self.l_eye_angle = _get_angle(self.l_eye)


def align(template, img, img_points):
    img_corners = _extract_l_eye_corners(img_points)

    img = _scale_img(template, img, img_corners)
    img = _rotate_img(template, img, img_corners)
    img = _translate_img(template, img, img_corners)

    return resizeAndPad(img, RESIZE_H_W)


# https://stackoverflow.com/a/44659589
def resizeAndPad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method:
    # shrinking image
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    # stretching image
    else:
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w)/h
    saspect = float(sw)/sh

    if (saspect > aspect) or ((saspect == 1) and (aspect <= 1)):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0

    else:   # (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(pad_color, (list, tuple, np.ndarray)):
        # color image but only one color provided
        pad_color = [pad_color] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=pad_color)

    return scaled_img


def _get_angle(corners) -> float:
    dx = corners[1].x - corners[0].x
    dy = corners[1].y - corners[0].y
    return math.degrees(math.atan2(dy, dx))


def _get_scale(template, img_corners):
    img_corner_l = img_corners[0]
    img_corner_r = img_corners[1]
    return math.dist(template.l_eye[0], template.l_eye[1]) / math.dist(img_corner_l,
                                                                       img_corner_r)


def _scale_img(template, img, img_points):
    scale = _get_scale(template, img_points)
    if scale < 1:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _rotate_img(template, img, img_corners):
    # calc angle of template, calc angle of img, rotate accordingly
    img_angle = _get_angle(img_corners)
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2,rows/2), template.l_eye_angle - img_angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols,rows))


def _translate_img(template, img, img_corners):
    transformation_matrix = np.float32([[1,0,template.l_eye[0].x-img_corners[0].x],
                                        [0,1,template.l_eye[0].y-img_corners[0].y]])
    rows, cols = img.shape[:2]
    return cv2.warpAffine(img, transformation_matrix, (cols,rows))


def _extract_l_eye_corners(points) -> [Coord, Coord]:
    return [Coord(points[36].x, points[36].y),
            Coord(points[39].x, points[39].y)]


if __name__ == '__main__':
    pass
