import numpy as np


def generateAverageFace(warped_imgs):
    """
    Generates the average face by averaging the pixel intensities of the warped images.

    Args:
        warped_imgs (list): List of warped images.

    Returns:
        numpy.ndarray: The generated average face.
    """
    return np.round(np.mean(np.array(warped_imgs), axis=0)).astype(np.uint8)
