import cv2
import numpy as np
import matplotlib.pyplot as plt

import pysift
import logging
logger = logging.getLogger(__name__)

flag = int(input("For image recognition task choose 1\nFor image allignment task choose 2\n:"))

if flag==1:
    #object recognition task
    img2 = cv2.imread("Beats-Rhymes-and-Life-Cover.jpg",0) # trainImage
    img1 = cv2.imread("tribe_cropped.jpg",0) # queryImage

    img2_orig = img2
    img1_orig = img1


if flag==2:
    img2 = cv2.imread("Beats-Rhymes-and-Life-Cover.jpg",0) 
    rows,cols = img2.shape
    #downscaling the img (for faster results)
    img2 = cv2.resize(img2,(rows//2,cols//2))
    rows,cols = img2.shape

    M = np.float32([[1/2,0,50],[0,1/2,50]])
    img1 = cv2.warpAffine(img2,M,(cols,rows))



MIN_MATCH_COUNT = 10

#MAIN ALGORITHM------------------------------------------------
# Compute SIFT keypoints and descriptors
kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

#normalize the descriptor vectors
def normalize_descriptors(des):
    for i in range(des.shape[0]):
        des[i,:] = des[i,:]/np.sqrt(np.dot(des[i,:],des[i,:]))

    return des

des1 = normalize_descriptors(des1)
des2 = normalize_descriptors(des2)



# Initialize and use FLANN
#USING THE K NEAREST NEIGHBORS ALGORITHM TO FIND THE MATCHES (K=2)
#we are trying to match each feature of the 2 images 
#For each feature (of the smallest image preferably) we are finding the top 2 features (of the other image) that minimizes the L2 norm
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
#for each match we test if the top first match is good by examing the ratio of the top 2 features.(If the ratio is close to 1 then
# the top 2 features that have been matched are so close to each other (probably outlier) and if that ratio is <0.7 then it is probably a good match)
#That fact comes from the fact that the metric that we use (L2 distance) is unhelpfull in high dimensions because all vectors are almost equidistant 
# to the search query vector (imagine multiple points lying more or less on a circle with the query point at the center; 
# the distance from the query to all data points in the search space is almost the same) CURSE OF DIMENSIONALITY
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)


if len(good) > MIN_MATCH_COUNT:
    # Estimate homography between template and scene
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

    # Draw detected template in scene image
    h, w = img1.shape
    pts = np.float32([[0, 0],
                      [0, h - 1],
                      [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    h1, w1 = img1.shape
    h2, w2 = img2.shape
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = int((h2 - h1) / 2)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

    for i in range(3):
        newimg[hdif:hdif + h1, :w1, i] = img1
        newimg[:h2, w1:w1 + w2, i] = img2

    # Draw SIFT keypoint matches and keep 3 of them for image alignment afterwards
    count = 0
    pts1 = []
    pts2 = []
    for m in good:

        count += 1
        pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
        pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
        cv2.line(newimg, pt1, pt2, (255, 0, 0))

        if (count>=10)and(count < 13):
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1]))
            pt2 = (int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1]))
            pts1 = np.concatenate( (pts1 , pt1))
            pts2 = np.concatenate( (pts2 , pt2))

    pts1 = np.reshape(pts1,(3,2))
    pts2 = np.reshape(pts2,(3,2))

else:
    print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
#-----------------------------------------------------------------------------------

#PLOTING--------------------------------------------------------------
def display_image_in_actual_size(img,title):

    dpi = 80
    im_data = img
    height, width = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    #add title
    ax.set_title(title)
    
#image alignment case
if flag==2:
    #using the homography (projective transform) that we estimated earlier to get back the transformed image (image1)
    l = cv2.warpPerspective(img1,M,(rows,cols))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    ax1.imshow(img2,cmap='gray')
    ax1.set_title("Initial image")
    ax2.imshow(img1,cmap='gray')
    ax2.set_title("Geometricaly transformed image")
    ax3.imshow(l,cmap="gray")
    ax3.set_title("Reconstruction after correspondance")
    f.tight_layout()
    plt.show()


if flag==1:
    display_image_in_actual_size(img2_orig,"Scene that contains the object")
    display_image_in_actual_size(img1_orig,"Query image")

    plt.figure()
    plt.imshow(newimg)
    plt.title("object recognition")
    plt.show()


a = 0