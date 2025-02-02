import cv2
import numpy as np


def computeDelaunayTriangles(avg_points, img_dim=600):
    """
    Compute Delaunay triangles for a set of points.

    Args:
        avg_points (ndarray (num_landmark, 2)) array of average points (landmarks).
        img_dim (int): Dimension of the image (default is 600).

    Returns:
        delaunay_triangles (numpy.ndarray): List of triangles.
        delaunay_indices (list): List of indices for each triangle in avg_points.
    """
    img_rect = (0, 0, img_dim, img_dim)
    subdiv = cv2.Subdiv2D(img_rect)

    for point in avg_points:
        subdiv.insert(point)

    avg_points = avg_points.astype(int)
    delaunay_triangles = subdiv.getTriangleList().astype(int)
    delaunay_indices = []

    for t in delaunay_triangles:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        indices = [np.where((avg_points == p).all(axis=1))[0][0] for p in pts]
        delaunay_indices.append(indices)

    return delaunay_triangles, delaunay_indices
