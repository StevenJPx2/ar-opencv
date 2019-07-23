import cv2
import numpy as np

MIN_MATCHES = 30
cap = cv2.imread('scene.jpg')    
model_video = cv2.VideoCapture(0)
"""
model_video.set(3, 640)
model_video.set(4, 480)
"""

# ORB keypoint detector
orb = cv2.ORB_create()              
# create brute force  matcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  
# Compute scene keypoints and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cap, None)

while True:
    # Compute model keypoints and its descriptors

    ret, model = model_video.read()
    kp_model, des_model = orb.detectAndCompute(model, None)  

    # Match frame descriptors with model descriptors
    matches = bf.match(des_model, des_frame)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # assuming matches stores the matches found and 
    # returned by bf.match(des_model, des_frame)
    # differenciate between source points and destination points
    src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # compute Homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

        # Draw a rectangle that marks the found model in the frame
    h, w, l = cap.shape
    #print(cap.shape)
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    # project corners into frame
    dst = cv2.perspectiveTransform(pts, M)  
    # connect them with lines
    img2 = cv2.polylines(model, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

    if len(matches) > MIN_MATCHES:
        # draw first 15 matches.
        frame = cv2.drawMatches(model, kp_model, cap, kp_frame,
                            matches[:MIN_MATCHES], 0, flags=2)
        # show result
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break 
    else:
        print ("Not enough matches have been found - %d/%d" % (len(matches),
                                                            MIN_MATCHES))

    
model_video.release()
cv2.destroyAllWindows()