import dlib
import cv2

# location of the model (path of the model).
MODEL_PATH = "shape_predictor_68_face_landmarks.dat"

# Now this line will try to detect all faces in an image either 1 or 2 or more faces
def get_points(img: cv2.imread, frontal_face_detector, face_landmark_detector) -> dlib.points:
    """
    If a single face is detected, return the 68 landmarks
    """
    # read img using opencv
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # now from the dlib we are extracting the method get_frontal_face_detector()
    # and assign that object result to frontalFaceDetector to detect face from the image with
    # the help of the 68_face_landmarks.dat model
    frontal_face_detector = dlib.get_frontal_face_detector()
    # Now the dlip shape_predictor class will take model and with the help of that, it will show
    face_landmark_detector = dlib.shape_predictor(MODEL_PATH)
    # try to detect all faces in an image either 1 or 2 or more faces
    # i only want one, though
    faces = frontal_face_detector(imageRGB, 0)
    # check that only 1 face detected
    if len(faces) != 1:
        return None
    face = faces[0]
    # dlib rectangle class will detect face so that landmark can apply inside that area
    face_rectangle_dlib = dlib.rectangle(int(face.left()), int(face.top()),
                                         int(face.right()), int(face.bottom()))
    # put landmark on detected face with the help of faceLandmarkDetector
    detected_landmarks = face_landmark_detector(imageRGB, face_rectangle_dlib)
    return detected_landmarks.parts()


if __name__ == '__main__':
    pass
