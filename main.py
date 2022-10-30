from pathlib import Path
import cv2
import dlib
from face_detect import get_points
from transform import Template, align
from video import create_video


MODEL_PATH = 'shape_predictor_68_face_landmarks.dat'
TEMPLATE_PATH = 'template.jpg'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'


def main():
    # assign object result to frontal_face_detector to detect face from the image with
    # the help of the 68_face_landmarks.dat model
    frontal_face_detector = dlib.get_frontal_face_detector()
    # Now the dlip shape_predictor class will take model and with the help of that, it will show
    face_landmark_detector = dlib.shape_predictor(MODEL_PATH)

    template = Template(TEMPLATE_PATH,
                        frontal_face_detector, face_landmark_detector)

    # iterate through all jpgs
    dropped_count = 0
    for i, p in enumerate(Path('input').glob('*.jpg')):
        print(f'processing picture {i}, dropped count: {dropped_count} ...')
        img = cv2.imread(str(p))
        ps = get_points(img, frontal_face_detector, face_landmark_detector)
        if ps == None:
            dropped_count += 1
            print('\tdropped picture!')
            continue
        cv2.imwrite(f'{OUTPUT_DIR}/{str(i-dropped_count)}.jpg', align(template, img, ps))

    print('Total dropped pictures:', dropped_count)
    create_video()


if __name__ == '__main__':
    main()
