# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 20:59:16 2020

@author: ferna

A set of ordered images (assumed to be taken from the same standing position in
horizontal succession from left to right) is given in a folder. This program 
will output a panorama of the stitched images.

It will first determine the keypoints and their descriptors of each image
using the SIFT algorithm and after that it will find the correspondences using 
Brute-Force Matching with ratio test. 

Once the matches are obtained, they will be used to calculate the homography matrix
for each pair of images.

The images are stitched around the center image, all of them warped to match
the center image's frame of reference (using the homographies)
and fitted in a blank canvas created beforehand.

INPUT:
    
    parentFolder: Path to folder where the result will be stored.
    
    imagePathExtension: Path extension from parentFolder to the folder where 
                        the images that are to be stitched are stored. 

OUTPUT:
    
    panorama: result of the stitching process



For improved performance:

Brute-Force Matching proves to be quite slow for big images. Implementing Flann
based matching could improve the performance for those cases.

The SURF algorithm could be considered as well to speed up the keypoint 
and descriptor search.


"""
import numpy as np
import cv2 as cv
# import time
from os import listdir
from os.path import isfile, join

MIN_MATCH_COUNT = 10
nKeypoints = 2000
ratioMatch = 0.75

# parentFolder = "D:/Galeria/"
# imgPathExtension = "imgs3/"

def parseArguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-pf', '--parentFolder', help='Input parent folder path.')
    parser.add_argument('-ipe', '--imagePathExtension', help='Input path extension for where the input images are located.')
    args = parser.parse_args()
    return args

def readImgsFromFolder(fPath):
    
    """
    This function reads all the files in a folder (given the path) and
    stores them in a list which is then returned.
    
    """
    
    imgs = []
    
    imgPath = [f for f in listdir(fPath) if isfile(join(fPath, f))]
    
    for i in range(len(imgPath)):
        file = cv.imread(join(fPath,imgPath[i]))
        if file is None:                                    # Unnecessary?
            print("Error: Unable to read an image file") 
            exit(-1)
        else:
            imgs.append(file)
    
    return imgs

def findKpDesSIFT(imgs):
    
    """
    This function detects keypoints and descriptors of each image using SIFT 
    algorithm and stores them in a list
    """
    
    
    sift = cv.SIFT_create(nKeypoints)
    gimgs = []
    imgsKpDes = []  # imgsKpDes[img#][kp,des][kp or des #]
    
    for i in range(len(imgs)):
        gimgs.append(cv.cvtColor(imgs[i],cv.COLOR_BGR2GRAY)) # Convert imgs to gray
        imgsKpDes.append(sift.detectAndCompute(gimgs[i],None)) # Compute kp and des
        
    
    
    return imgsKpDes

def findKpDesORB(imgs):
    
    """
    This function detects keypoints and descriptors of each image using ORB 
    algorithm and stores them in a list
    """
    
    orb = cv.ORB_create(nKeypoints)
    gimgs = []
    imgsKpDes = []  # imgsKpDes[img#][kp,des][kp or des #]
    
    for i in range(len(imgs)):
        gimgs.append(cv.cvtColor(imgs[i],cv.COLOR_BGR2GRAY)) # Convert imgs to gray
        imgsKpDes.append(orb.detectAndCompute(gimgs[i],None)) # Compute kp and des
        
        
    return imgsKpDes

def matchImgsBFM(imgsKpDes):
    
    """
    This function matches the keypoints from each pair of images using
    Brute Force matching and ratio test, and stores the results in a list.
    """
    
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    matches = []    # matches[img pair #][match object #]
    
        # modify ratioMatch for different results 
        # closer to 1 is +matches but +mistakes
                    
    
    # Iterate through the images and match with the next (no looping around)
    
    for i in range(len(imgsKpDes)-1):
        
        match = bf.knnMatch(imgsKpDes[i][1],imgsKpDes[i+1][1], k=2)
        
        good = []
        for m,n in match:
            if m.distance < ratioMatch*n.distance: 
                good.append(m)
                
        matches.append(good)
    
    return matches    
    
def findHomographies(imgsKpDes,matches):
    
    """
    This function computes the homography matrix between each pair of images,
    using RANSAC to get rid of outliers, and stores them in a dictionary.
    """
    
    keys = []   # Format Hij
    H_all = []
    
    for i in range(len(matches)):
    
        if len(matches[i])>MIN_MATCH_COUNT:
            src_pts = np.float32([ imgsKpDes[i][0][m.queryIdx].pt for m in matches[i] ]).reshape(-1,1,2)
            dst_pts = np.float32([ imgsKpDes[i+1][0][m.trainIdx].pt for m in matches[i] ]).reshape(-1,1,2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
           
            keys.append("H{}{}".format(i,i+1))
            H_all.append(M)
            
        else:
            print( "Not enough matches are found - {}/{}".format(len(matches[0]), MIN_MATCH_COUNT) )
            
    H_all = dict(zip(keys, H_all))
        
    return H_all

def compute_H_wrt_middle_img(H_all, middle_id):
    
    """
    This function computes the homography matrix with respect to the middle image
    for each of the homographies calculated before, and stores them in a dictionary.
    
    It adds the identity matrix as the transformation for the middle image.
    """

    # Hij is pts_in_img_j = Hij * pts_in_img_i
    # If num of images are 5, we have H01, H12, H23, H34 i.e
    # Pts_in_img_1 = H01 * pts_in_img_0
    # Pts_in_img_2 = H12 * pts_in_img_1
    # Pts_in_img_3 = H23 * pts_in_img_2
    # Pts_in_img_4 = H34 * pts_in_img_3

    # We need all the matrices wrt to the middle image frame of reference i.e H02, H12, H32, H42, H22

    # H02 = H12 * H01
    # H12 = H12


    num_imgs = len(H_all)+1

    key = "H{}{}".format(middle_id, middle_id)
    H_all[key] = np.eye(3)

    for i in range(0, middle_id):
        key = "H{}{}".format(i, middle_id)  # H02
        j = i
        temp = np.eye(3)
        while j < middle_id:
            key_t = "H{}{}".format(j, j+1)
            temp = np.matmul(H_all[key_t], temp)
            j += 1

        H_all[key] = temp


    # H32 = inv(H23)
    # H42 = inv(H23) * inv(H34)
    for i in range(middle_id+1, num_imgs):
        key = "H{}{}".format(i, middle_id)  # H32

        temp = np.eye(3)

        j = i-1

        while j >= middle_id:
            key_t = "H{}{}".format(j, j+1)
            temp = np.matmul(np.linalg.inv(H_all[key_t]), temp)
            j -= 1

        H_all[key] = temp


    return H_all

def findMiddleId(input_list):
    
    """
    This function finds the center index in a list of elements
    """
    
    middle = float(len(input_list))/2
    if middle % 2 != 0:
        return int(middle - .5)
    else:
        return int(middle-1)

def get_blank_canvas(H_all,imgs):

    """
    This function creates a blank canvas to fit all the images transformed with
    respect to the center image
    """

    middleId = findMiddleId(imgs)

    img_h, img_w, _ = imgs[0].shape

    min_crd_canvas = np.array([np.inf, np.inf, np.inf])
    max_crd_canvas = np.array([-np.inf, -np.inf, -np.inf])
    
    # middleId = findMiddleId(imgs)

    for i in range(len(imgs)):
        key = "H{}{}".format(i, middleId)
        H = H_all[key]
        min_crd, max_crd = compute_extent(H, img_w, img_h)

        min_crd_canvas = np.minimum(min_crd, min_crd_canvas)
        max_crd_canvas = np.maximum(max_crd, max_crd_canvas)

    width_canvas = np.ceil(max_crd_canvas - min_crd_canvas)[0] + 1
    height_canvas = np.ceil(max_crd_canvas - min_crd_canvas)[1] + 1

    canvas_img = np.zeros((int(height_canvas), int(width_canvas), 3), dtype=np.int64)

    offset = min_crd_canvas.astype(np.int64)
    offset[2] = 0  # [x_offset, y_offset, 0]

    mask = np.ones((int(height_canvas), int(width_canvas)))

    return canvas_img, mask, offset
    
def compute_extent(H, img_w, img_h):

    """
    This function returns the max and min coordinates of an image once it is
    transformed to the center image's frame of reference.
    """        

    corners_img = np.array([[0, 0], [img_w, 0], [img_w, img_h], [0, img_h]])

    t_one = np.ones((corners_img.shape[0], 1))
    t_out_pts = np.concatenate((corners_img, t_one), axis=1)
    canvas_crd_corners = np.matmul(H, t_out_pts.T)
    canvas_crd_corners = canvas_crd_corners / canvas_crd_corners[-1, :]  # cols of [x1, y1, z1]

    min_crd = np.amin(canvas_crd_corners.T, axis=0)  # [x, y, z]
    max_crd= np.amax(canvas_crd_corners.T, axis=0)
    
    return min_crd, max_crd

def get_pixel_coord(mask):
    """        
    This function returns the homogeneous coordinates of the white pixels in the mask
    """
    y, x = np.where(mask)
    pts = np.concatenate((x[:,np.newaxis], y[:, np.newaxis], np.ones((x.size, 1))), axis=1) # rows of [x1, y1, 1]

    return pts

def fit_image_in_target_space(img_src, img_dst, mask, H, offset=np.array([0, 0, 0])):
    """
    This function
    """

    pts = get_pixel_coord(mask)  # rows of [x1, y1, 1]

    pts = pts + offset

    out_src = np.matmul(H, pts.T)  # out_src has cols of [x1, y1, z1]

    out_src = out_src/out_src[-1,:]

    # Return only x, y non-homogenous coordinates
    out_src = out_src[0:2, :]  # corresponds to pixels in img_src
    out_src = out_src.T  # rows of [x1, y1]

    # Convert pts to out_src convention
    pts = pts[:, 0:2].astype(np.int64)  # Corresponds to pixel locs in img_dst, rows of [x1,y1]

    h, w, _ = img_src.shape

    get_pixel_val(img_dst, img_src, pts, out_src, offset)

    return img_dst

def get_pixel_val(img_dst, img_src, pts, out_src, offset):
    """
    :param img_dst:
    :param pts: pts for img_dst rows of [x1, y1]
    :param out_src: rows of [x1, y1], corresponding pts in src img after homography on dst points
    :return:
    """
    h, w, _ = img_src.shape
    tl = np.floor(out_src[:, ::-1]).astype(np.int64) # reverse cols to get row, col notation
    br = np.ceil(out_src[:, ::-1]).astype(np.int64)

    pts = pts - offset[:2]

    r_lzero = np.where(~np.logical_or(np.any(tl < 0, axis=1), np.any(br < 0, axis=1)))
    pts = pts[r_lzero[0], :]
    out_src = out_src[r_lzero[0], :]
    tl = tl[r_lzero[0], :]
    br = br[r_lzero[0], :]

    r_fl = np.where(~np.logical_or(tl[:, 0] >= h-1, tl[:, 1] >= w-1))
    pts = pts[r_fl[0], :]
    out_src = out_src[r_fl[0], :]
    tl = tl[r_fl[0], :]
    br = br[r_fl[0], :]

    r_ce = np.where(~np.logical_or(br[:, 0] >= h-1, br[:, 1] >= w-1))
    pts = pts[r_ce[0], :]
    out_src = out_src[r_ce[0], :]
    tl = tl[r_ce[0], :]
    br = br[r_ce[0], :]

    tr = np.concatenate((tl[:, 0:1], br[:, 1:2]), axis=1)

    bl = np.concatenate((br[:, 0:1], tl[:, 1:2]), axis=1)

    weight = np.zeros((out_src.shape[0], 4))

    weight[:, 0] = np.linalg.norm(tl-out_src[:, ::-1], axis=1)
    weight[:, 1] = np.linalg.norm(tr-out_src[:, ::-1], axis=1)
    weight[:, 2] = np.linalg.norm(bl-out_src[:, ::-1], axis=1)
    weight[:, 3] = np.linalg.norm(br - out_src[:, ::-1], axis=1)

    weight[np.all(weight == 0, axis=1)] = 1  # For entries where they exactly overlap
    weight = 1/weight

    # pts = pts - offset[:2]

    img_dst[pts[:,1], pts[:,0], :] = (img_src[tl[:,0], tl[:,1], :] * weight[:, 0:1] + \
                                     img_src[tr[:,0], tr[:,1], :] * weight[:, 1:2] + \
                                     img_src[bl[:,0], bl[:,1], :] * weight[:, 2:3] + \
                                     img_src[br[:,0], br[:,1], :] * weight[:, 3:4])/ np.sum(weight, axis=1, keepdims=True)


    return img_dst

def stitch(H_all, imgs, parentFolder):
    
    """
    This function uses other functions in the program to create the canvas and
    fit each image within it. Finally, it writes the results in the destination in
    the parent folder.
    """
    
    canvas_img, mask, offset = get_blank_canvas(H_all,imgs)
    middle_id = findMiddleId(imgs)
    
    for i, img in enumerate(imgs):

        key = "H{}{}".format(i, middle_id)
        H = H_all[key]

        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        canvas_img = fit_image_in_target_space(img_rgb, canvas_img, mask, np.linalg.inv(H),
                                               offset=offset)  # the inp to fit_image_in_target_space
        # pts_in_img_2 = H * pts_in_canvas
        mask[np.where(canvas_img)[0:2]] = 0

        result_path = join(parentFolder, 'panorama.jpg')
        
    cv.imwrite(result_path, canvas_img[:, :, (2, 1, 0)])
        

def main():
    
    args = parseArguments()
    
    parentFolder = args.parentFolder
    imgPathExtension = args.imagePathExtension
    

    # Read images from a folder and stores them in a list

    imgs = readImgsFromFolder(join(parentFolder, imgPathExtension))
    
    # Finds the keypoints and descriptors of each image
    
    imgsKpDes = findKpDesSIFT(imgs) # Using SIFT algorithm
    # imgsKpDes = findKpDesORB(imgs) # Using ORB algorithm
    
        
    # Computes the correspondences between each pair of images
    
    matches = matchImgsBFM(imgsKpDes) # Using Brute-Force matching
    # matches = matchImgsFlann(imgsKpDes) # Using Flann based matching (not working)
    
    
    # Show the matches
    """
    for i in range(len(matches)):
        
        filename = "imgMatches" + str(i) + ".jpg"
        
        imgMatches = cv.drawMatches(imgs[i],imgsKpDes[i][0],
                                       imgs[i+1],imgsKpDes[i+1][0],matches[i],
                                       None,flags=2)
        
        cv.imwrite(join(parentFolder,filename),imgMatches)
    """
    
    # Finds the homographies between each pair of images (H01, H12, ...)
    
    M = findHomographies(imgsKpDes,matches)
    
    # Converts the homographies to the center image's frame of reference
    
    H = compute_H_wrt_middle_img(M,findMiddleId(imgs))
    
    # Stitches the given images using the homographies computed
    
    stitch(H, imgs, parentFolder)
    
    
if __name__ == "__main__":
    main()