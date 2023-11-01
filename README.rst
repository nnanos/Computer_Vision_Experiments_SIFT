=======================================================================
Computer Vision Experiments on image alignment and object detection tasks with SIFT 
=======================================================================

In this repo I do some Experiments on image alignment and object detection tasks.
These tasks can be reduced to a basic problem on computer vision, the
correspondence problem i.e. given two images find the geometric transformation
that connects them. This problem is solved either with feature based techniques
or with correlation based techniques. In this repo, the problem will be approached
with feature based techniques. More specifically we will use SIFT 
(SCALE INVARIANT FEATURE TRANSFORM) transform which is described in the paper
on the reference section. This Transform is used to get some good features of
the examined images. 


General points of the SIFT algorithm that are worth noting:
=======================================================================



**Correspondence algorithm implemented:**

#. Apply the SIFT algorithm to the two input images and get the descriptors
   and the corresponding keypoints for each one.

#. Apply the k-nn (k-nearest neighbor) algorithm with k=2 for the two sets 
   of descriptors that arose in the previous step. Since k=2 was chosen,
   the algorithm for each descriptor of one image will find the two closest
   (in L2 sense) descriptors to the other image .

#. In this step we are going to discard some of the matches generated in
   previous procedure. According to lowe's paper for each match (which
   consists of the 2 best) we take the ratio of the first best to the second
   best distance and if this is less than 0.7 then we keep the first one
   best match as "good" otherwise we don't keep it (see explanation below).


An explanation for the third step : This need for rejection arises from the
weakness of the L2 distance in high dimensional vectors. Two descriptors may
be orthogonal and give a distance of 2 and "corresponding" descriptors to have
a value close to two (CURSE OF DIMENSIONALITY).



**RANSAC Algorithm Description:**

#. We randomly sample a number of samples (eg, corresponding points) of the dataset.
   This number is the minimum so that we can estimate the parameters (model)
   of our problem (eg 4 corresponding points to estimate a projective Transform).

#. We assume that the selected data are inliers (eg correct corresponding points)
   and we use them to estimate our model.

#. We count how many of the remaining data points (eg point mappings) agree with
   the model and this constitutes the score of the model. If the current model
   has the best score then we keep this one and if not then we keep the
   previous best model.

#. We repeat the previous steps for a number of N iterations or terminate
   earlier if the model has reached a predetermined tolerance error .


The exact number of iterations for the algorithm to converge (ie produce a model
which has not relied on outliers ) is proven by probability theory to be given by the following relation:

.. Image:: /Documentation_Images/Ransac_relation.png  


**Adaptation of RANSAC to the correspondence problem.**
In our problem RANSAC is used to estimate the homography (projective
transform) which relates the two input images of the correspondence algorithm.
More specifically here the dataset we sample from is the "good" matches
(corresponding points) and the output of the algorithm is a projective
transformation which with high probability has been estimated from
correct corresponding points (inliers).



Experiments
============

Below are two examples in which the correspondence algorithm is used.
Problems like **object detection** and **image alignment** for which the
RANSAC algorithm was also used.



* Image Alignment


.. Image:: /Documentation_Images/Image_Alignment.png  


* Object Detection


.. Image:: /Documentation_Images/Object_Det.png   


However for the last problem (for the specific inputs) the feature based
method which we used is overkill as we could use a more simplified method 
e.g. template matching with Laplacian pyramid (as implemented in
the first work with the pyramids). For more complex inputs where the object
we are looking for is related to someone complex projective transformation
with the object in the scene which we are looking for is necessary that 
we use a feature based method like we use here.

Reproduce the Experiments
============




References
============


Free software: MIT license
============
