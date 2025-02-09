from average_face import alignImages
from average_face import computeDelaunayTriangles
from average_face import drawLandmarks
from average_face import detectFace, getLandmarks
from average_face import generateAverageFace
from average_face import loadImages
from average_face import normalizeImage

import cv2


def main():
    # Load images
    image_paths = loadImages("images/*")
    normalized_imgs = []

    for img_path in image_paths:
        img = cv2.imread(img_path)
        face = detectFace(img)

        if face is None:
            print(f"Error: Couldn't detect any faces in {img_path}")
        else:
            print(f"Processing: {img_path}")
            landmarks = getLandmarks(img, face)
            normalized_img, updated_landmarks = normalizeImage(img, landmarks)
            normalized_imgs.append((normalized_img, updated_landmarks))

    # Compute average landmarks
    avg_points = (sum([landmarks for img, landmarks in normalized_imgs]) / len(normalized_imgs))

    # Delaunay Triangulation
    delaunay_triangles, delaunay_indices = computeDelaunayTriangles(avg_points)
    avg_points = avg_points.astype(int)

    # Align Images
    warped_imgs = alignImages(normalized_imgs, delaunay_indices, avg_points)

    # Generate Average Face
    final_img = generateAverageFace(warped_imgs)

    # Display the final result
    cv2.imshow("Average Face", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
