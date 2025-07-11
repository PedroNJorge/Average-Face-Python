# Average Face Computation in Python
This project generates an average face from a given set of images by detecting facial landmarks, aligning features, and blending the images using Delaunay Triangulation.

# **Import Libraries**
Throughout this project we will use:

* **OpenCV**
* **Dlib**
* **NumPy**
* **Pillow** (will be used to help display images in Jupyter Notebook)


# **Extract facial features**<br>
<div align="center">
  <img src="https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width=400 height=300>
</div>

With the help of **dlib**, we will extract the facial features of each image, as demonstrated in the picture above. We will use **get_frontal_face_detector** to detect where the face is, and **shape_predictor_68_face_landmarks.dat** to extract the 68 landmarks of the face. Here is an example:<br>

<div align="center">
  <img src="https://github.com/user-attachments/assets/39629786-3a29-4d9d-b574-e2b8763571e4">
</div>


# **Normalize images**
In drawing, faces can be simplified using the “Rule of Thirds,” which divides the face into horizontal and vertical thirds to help position facial features accurately.<br><br>

**Rule of Thirds:** The face can be divided into three equal horizontal sections. The top third is from the hairline to the eyebrows, the middle third is from the eyebrows to the bottom of the nose, and the bottom third is from the bottom of the nose to the chin. This method helps ensure that facial features are proportionally placed.<br>
**Facial Features Placement:** Eyes are typically positioned halfway between the top of the head and the chin, which aligns with the horizontal third division. The nose line is found in the middle of the eye line and the bottom of the chin, and the mouth line is about one-third of the way down from the nose line to the chin.<br><br>

Since each image has a different size, we need to normalize them, warping each to a **600x600 image**. With the drawing rules stated earlier, let's define where certain facial features will warp to:<br><br>

* Left corner of the **left eye**: (180, 200)
* Right corner of the **right eye**: (420, 200)
* **Bottom lip**: (300, 400)

Now that we know the starting and ending positions, we can use the similarity transform (rotation, translation and scale). To find this transformation, we will use `cv2.getAffineTransform`. With this matrix, we can warp the image with `cv2.warpAffine` and update the landmarks:<br>

<div align="center">
  <img src="https://learnopencv.com/wp-content/ql-cache/quicklatex.com-b6e614b5448854f2c83abcb6e5786774_l3.png">
</div>

The *first* and *second* columns **rotate and scale** the vector. You will need to add the *last* column, which represents the **translation**.


# **Delaunay Triangulation**
Now that the images are normalized, we need to align the rest of the remaining features. To do that, we will first calculate the **Delaunay Triangulation** of the final image:<br>

* Calculate the **landmarks** of the final image, by taking the **average** of the images' landmarks;
* Create a **planar subdivision**, with `cv2.Subdiv2D`, that contains the landmarks of the final image;
* Extract the **Delaunay triangles** with `cv2.getTriangleList` and the respective **indices of the landmarks** used to form each triangle.

<div align="center">
  <img src="https://github.com/user-attachments/assets/03b65f05-2b76-4805-a2c6-b09221ed9e4a">
</div>


# **Align images**
With the Delaunay Triangulation of the final image already computed, our next step is to warp the Delaunay Triangles of each input image so that their landmarks align with those of the final image. To warp, we will use `cv2.getAffineTransform` again. However, `cv2.warpAffine` applies transformations to the entire image, while we only need to warp the pixels within the triangle.<br>
To solve that, we will create a mask for the triangle with the help of `cv2.fillConvexPoly` (this will generate a black image with the triangle painted in white).


# **Generate Average Face**
Now that we have everything set up, we can generate the average face by taking the **mean pixel intensity of all warped images**. Below is the final output generated by averaging the faces of **Elon Musk**, **Donald Trump**, and **Mark Zuckerberg**.

<div align="center">
  <img src="https://github.com/user-attachments/assets/038e2dfb-bafd-4c1a-b283-68838019a0af">
</div>
