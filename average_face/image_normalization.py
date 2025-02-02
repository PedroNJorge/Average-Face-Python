import cv2
import numpy as np


LEFT_EYE_CORNER = 36
RIGHT_EYE_CORNER = 45
BOTTOM_LIP = 57


def normalizeImage(img, landmarks, img_dim=600):
    """
    Warps the image input to a 600x600 image with the aid of the outer corner of the eyes and the bottom lip 
    
    Args:
      img (ndarray (width, height, 3)): Image
      landmarks (dlib_full_object_detection (68)): Facial landmarks of the image
      img_dim (int): Dimension of the image (default is 600).      
    Returns
        (normalized_img, updated_landmarks) (ndarray (600, 600, 3), ndarray (68, 2)): Warped 600x600 image
                                                                                        and landmarks for the warped image
    """
    left_eye = landmarks.part(LEFT_EYE_CORNER)
    right_eye = landmarks.part(RIGHT_EYE_CORNER)
    bot_lip = landmarks.part(BOTTOM_LIP)

    # Create source array with float32 type
    src = np.array([[left_eye.x, left_eye.y],
                    [right_eye.x, right_eye.y],
                    [bot_lip.x, bot_lip.y]], dtype=np.float32)
    
    # Create destination array with float32 type
    dst = np.array([[180, 200],
                    [420, 200],
                    [300, 400]], dtype=np.float32)

    # Get transformation matrix
    T = cv2.getAffineTransform(src, dst)

    # Warp Image
    normalized_img = cv2.warpAffine(img, T, (img_dim,img_dim))

    # Update landmarks (T*landmarks + translation), but first convert to ndarray
    npLandmarks = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(68)], dtype=np.float32)
    last_row = T[:, -1]
    translation = np.array([last_row for _ in range(68)], dtype=np.float32)
    
    updated_landmarks = (np.delete(T, -1, axis=1) @ npLandmarks.transpose()).transpose() + translation

    # Add more points to help with alignment later on
    updated_landmarks = np.vstack((updated_landmarks,   [[0, 0],
                                                        [300, 0],
                                                        [600-1, 0],
                                                        [0, 300],
                                                        [600-1, 300],
                                                        [0, 600-1],
                                                        [300, 600-1],
                                                        [600-1, 600-1]]))

    return (normalized_img, updated_landmarks.astype(int))
