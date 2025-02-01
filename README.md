# Average-Face-Python (Not finished yet)
Generate an average face given numerous images of people
# **Import Libraries**

Throughtout this project we will use:

* **OpenCV**
* **Dlib**
* **NumPy**
* **Pillow** (will be used to help display images here in Jupyter Notebook)

# **Exctract facial features**<br>

<div align="center">
  <img src="https://pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup.jpg" width=400 height=300>
</div> <br>

With the help of **dlib**, we will extract the facial features of each image, as demonstraded in the picture above. We will use **get_frontal_face_detector** to detect where the face is, and **shape_predictor_68_face_landmarks.dat** to extract the 68 landmarks of the face:<br>

<div style="display: flex; justify-content: space-between; align-items: center;">
    <img src="https://cdn.discordapp.com/attachments/1027187677403029575/1332853023982092328/Screenshot_from_2025-01-25_23-09-41.png?ex=6796c39c&is=6795721c&hm=f73d98a5358d6dc37334a35abbb19fa9560d9d85cc26a8313599d659f5404ad2&" alt="Image 1" style="width: 50%; margin-right: 10px;" />
    <img src="https://cdn.discordapp.com/attachments/1027187677403029575/1332853066096971806/Screenshot_from_2025-01-25_23-10-20.png?ex=6796c3a6&is=67957226&hm=9fc02936c120d620f4c82fe383a3e2b1cf22d55e8bd7dcc0332be1dc84ec1d69&" alt="Image 3" style="width: 50%;" />
</div><br>

# **Normalize images**

In drawing, faces can be simplified using the “Rule of Thirds,” which divides the face into horizontal and vertical thirds to help position facial features accurately.<br><br>

**Rule of Thirds:** The face can be divided into three equal horizontal sections. The top third is from the hairline to the eyebrows, the middle third is from the eyebrows to the bottom of the nose, and the bottom third is from the bottom of the nose to the chin. This method helps ensure that facial features are proportionally placed.<br>
**Facial Features Placement:** Eyes are typically positioned halfway between the top of the head and the chin, which aligns with the horizontal third division. The nose line is found in the middle of the eye line and the bottom of the chin, and the mouth line is about one-third of the way down from the nose line to the chin.<br><br>

Since each image has variant size, we need to normalize them, warping each to a **600x600 image**. With the drawing rules stated earlier, let's define where certain facial features will warp to:<br><br>

* Left corner of the **left eye**: (180, 200)
* Right corner of the **right eye**: (420, 200)
* **Bottom lip**: (300, 400)

Now that we know the starting and ending positions, we can use the similarity transform (rotation, translation and scale). To find this transformation, we will use **getAffineTransform**. With this matrix, we can warp the image with **warpAffine** and update the landmarks:<br>

<div align="center">
  <img src="https://learnopencv.com/wp-content/ql-cache/quicklatex.com-b6e614b5448854f2c83abcb6e5786774_l3.png" width=400 height=300>
</div> <br>

The *first* and *second* columns **rotate and scale** the vector. You will need you to add the *last* column, that represents the **translation**.

# **Delaunay Triangulation**

Now that the images are normalized, we need to allign the rest of the remaining features. To do that, we will first calculate the **Delaunay Triangulation** of the final image:<br>

* Calculate the **landmarks** of the final image, by taking the **average** of the images' landmarks;
* Create a **planar subdivision**, with **Subdiv2D**, that contains the landmarks of the final image;
* Extract the **delaunay triangles** with **getTriangleList** and the respective **indices of the landmarks** used to form each triangle.
# image here...

# **Align images**

We already have the delaunay triangulation of the final image. Now, before we can produce the final image, we will warp each image's delaunay triangles, such that their landmarks correspond to the landmarks of the final image. To warp, we will use **getAffineTransform** again, but this time we have a problem: **warpAffine** warps the whole image, but we only want to warp the triangle.<br>
To solve that, we will create a mask for the triangle with the help of **fillConvexPoly** (this will generate a black image with the triangle painted in white).
# image here...


# **Generate Average Face**

Now that we have everything setup, we can generate the average face by taking the **mean of the warped images' pixel intensity**:

# final image
