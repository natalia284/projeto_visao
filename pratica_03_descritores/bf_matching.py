import cv2
import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt

def match_images(im1, im2, kp1, kp2, des1, des2, n_show=20, use_knn=False):
    # Create BFMatcher object
    bf = cv.BFMatcher()

    if not use_knn:
        # Match descriptors
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first matches
        img_match = cv.drawMatches(im1, kp1, im2, kp2, matches[:n_show], None,
                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        matches = bf.knnMatch(des1, des2, k=2)
        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])

        img_match = cv.drawMatchesKnn(im1, kp1, im2, kp2, good[:n_show], None,
                                      flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_match


def bf_orb(im1, im2, n_show=20, use_knn=False):
    orb = cv.ORB_create()

    # Find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    return match_images(im1, im2, kp1, kp2, des1, des2, n_show=n_show, use_knn=use_knn)


def bf_surf(im1, im2, n_show=20, use_knn=False):
    # Check if SURF is available (it is patented)
    if hasattr(cv, 'xfeatures2d'):
        # Create the SURF detector
        surf = cv.xfeatures2d.SURF_create(hessianThreshold=400)

        # Find the keypoints and descriptors with SURF
        kp1, des1 = surf.detectAndCompute(im1, None)
        kp2, des2 = surf.detectAndCompute(im2, None)

        return match_images(im1, im2, kp1, kp2, des1, des2, n_show=n_show, use_knn=use_knn)
    else:
        print("SURF not available in your OpenCV version.")
        return None


############


img1 = cv.imread("/home/natalia/Música/praticas_visão/imagens/sub_imagens/antartica.jpg")
img2 = cv.imread("/home/natalia/Música/praticas_visão/imagens/sub_imagens/antartica_lata.jpg")


# SURF
im_surf = bf_surf(img1, img2, use_knn=True)

# ORB
im_orb = bf_orb(img1, img2, use_knn=True)

# Plot the results
plt.subplot(121).set_ylabel("SURF"), plt.imshow(im_surf, 'gray') if im_surf is not None else print("SURF image not generated")
plt.subplot(122).set_ylabel("ORB"), plt.imshow(im_orb, 'gray')

plt.show()
