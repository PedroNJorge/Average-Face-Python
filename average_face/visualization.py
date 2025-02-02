import cv2


RED = (0, 0, 255)
GREEN = (0, 255, 0)


def drawLandmarks(img_lst, radius):
    """
    Display on screen the images with it's landmarks drawn on it
    
    Args:
      img_lst (img, landmarks):
          img (ndarray (width, height, 3)): Image
          landmarks (dlib_full_object_detection (68)): Facial landmarks of the image
      radius (int): Radius of the circle drawn for each landmark

    Returns
        None: Displays images
    """
    for img, landmarks in img_lst:
        # Draw a circle for each landmark
        for x, y in landmarks:
            cv2.circle(img, (x, y), radius, GREEN, -1)
