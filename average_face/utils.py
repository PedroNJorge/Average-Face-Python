import cv2
import glob


def loadImages(path="images/*"):
    """
    Loads images from a directory.

    Args:
        path (str): Filepath pattern to load images.

    Returns:
        list: List of image file paths.
    """
    return glob.glob(path)
