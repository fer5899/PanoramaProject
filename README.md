# Panorama Project

A Python program developed as a final assignment for the Computer Vision class at Brno University of Technology. This program takes a set of ordered images with some degree of horizontal overlap and merges them to create a panorama image.

## Usage

The program can be executed from the command line by providing two arguments:
- `-rp` or `--resultPath`: path to the folder where the panorama image will be stored.
- `-ip` or `--imagesPath`: path to the folder where the input images are stored.

Example:
```
python PanoramaProject.py -rp result_folder -ip images_folder
```

## How it works

The program uses the SIFT algorithm to determine the keypoints and their descriptors of each image. Then, it employs Brute-Force Matching with ratio test to find the correspondences between the images. 

The matches are then used to calculate the homography matrix for each pair of images. This homography matrix represents the transformation that needs to be applied to each image so that it aligns with the center image. 

The program then warps each image to match the center image's frame of reference using the calculated homographies, and fits all the images into a blank canvas created beforehand. 

## Performance optimization

Brute-Force Matching can be slow for large images. To improve performance, consider using Flann-based matching instead. Additionally, the SURF algorithm could be considered to speed up the keypoint and descriptor search.

## What I learned

Through this project, I improved my understanding of image processing using OpenCV and honed my skills in data management with Numpy. Additionally, I gained valuable experience working with computer vision techniques and algorithms. 

A detailed explanation of the project, including the algorithms and techniques used, can be found in the `PanoramaStitchingReport` pdf in the repository.


