import cv2
import numpy as np


def alignImages(normalized_imgs, delaunay_indices, avg_points, img_dim=600):
    """
    Aligns images by applying Delaunay triangulation to each image's landmarks.

    Args:
        normalized_imgs (list): List of (img, landmarks) tuples.
        delaunay_indices (list): List of Delaunay triangles' indices.
        avg_points (ndarray (num_landmarks,2)): Average points of landmarks.
        img_dim (int): Dimension of the image (default is 600).

    Returns:
        warped_imgs (list): List of aligned images.
    """
    warped_imgs = []

    for img, landmarks in normalized_imgs:
        landmarks = landmarks.astype(np.float32)

        # Extract corresponding triangles
        triangles_src = np.array([[landmarks[i] for i in triangle] for triangle in delaunay_indices], dtype=np.float32)
        triangles_dst = np.array([[avg_points[i] for i in triangle] for triangle in delaunay_indices], dtype=np.float32)

        warped_img = np.zeros((img_dim, img_dim, 3), dtype=img.dtype)

        for src, dst in zip(triangles_src, triangles_dst):
            # Get the affine transform
            T = cv2.getAffineTransform(src, dst)

            # Create a mask for the triangle in the destination image
            mask = np.zeros((img_dim, img_dim), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst), 255)

            # Warp the triangle
            warped_triangle = cv2.warpAffine(img, T, (img_dim, img_dim), borderMode=cv2.BORDER_REFLECT_101)

            # Blend the warped triangle into the final image
            warped_img[mask == 255] = warped_triangle[mask == 255]

        warped_imgs.append(warped_img)

    return warped_imgs
