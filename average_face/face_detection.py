import dlib

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def detectFace(img):
    """
    Detect the face present in the image. If the image contains two or more faces, only return the largest face.  

    Args:
      img (ndarray (width, height, 3)): Image

    Returns
        face (dlib_rectangles): Rectangle that delimits the face
    """
    face = detector(img, 1)

    if len(face) == 0:
        return None
    elif len(face) == 1:
        return face[0]
    else:
        return max(face, key=lambda rect: rect.width()*rect.height())


def getLandmarks(img, face):
    """
    Extracts 68 facial landmarks from the detected face.

    Args:
        img (numpy.ndarray): Image in BGR format.
        face (dlib.rectangle): Bounding box of detected face.

    Returns:
        numpy.ndarray: (68,2) array of facial landmark coordinates.
    """
    landmarks = predictor(img, face)
    return landmarks
