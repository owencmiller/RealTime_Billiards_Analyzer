import numpy as np
import cv2

def rotate_crop(image):
    h,w = image.shape[:2]
    center = (w/2, h/2)
    angle = 180
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (w, h))
    image = image[150:1000, 100:1850] # Do cropping
    return image

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def circle_detect(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=130, param2=15, minRadius=13, maxRadius=23)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Apply circles to overall image
            cv2.circle(img, (a,b), r, (0, 255, 0), 2)
            cv2.circle(img, (a,b), 1, (255, 0, 0), 3)
    return img
    
def pocket_detect(image):
    img = image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.medianBlur(gray, 5)
    gray_blurred_inv = cv2.bitwise_not(gray_blurred)

    ret, thresh = cv2.threshold(gray_blurred_inv, 220, 255, cv2.THRESH_BINARY) # <--- Try different values here
    # edges = cv2.Canny(thresh,100,200)
    detected_circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1.5, 100, param1=130, param2=15, minRadius=27, maxRadius=40)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            
            cv2.circle(img, (a,b), r, (255, 0, 0), 2)
            cv2.circle(img, (a,b), 1, (0, 255, 0), 3)
    return img

def table_detect(image):
    # assign red channel to zeros
    image[:,:,0] = np.zeros([image.shape[0], image.shape[1]])
    image[:,:,1] = np.zeros([image.shape[0], image.shape[1]])
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # define range of red color in HSV
    lower_red = np.array([0, 250, 250])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange (hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask.copy(),
                            cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    # if len(contours) > 0:
    i = 0
    for contour in contours:
        if i > 3:
            break
        red_area = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(red_area)
        cv2.rectangle(image,(x, y),(x+w, y+h),(0, 255, 0), 2)
        i += 1
    cv2.imshow('frame',image)
    # cv2.imshow('mask',mask)

def table_detect_2(image):
    # image[:,:,0] = np.zeros([image.shape[0], image.shape[1]])
    # image[:,:,1] = np.zeros([image.shape[0], image.shape[1]])

    image = cv2.medianBlur(image, 15)

    # red color boundaries [B, G, R]
    lower = [0, 0, 180]
    upper = [120, 120, 255]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    ret,thresh = cv2.threshold(mask, 40, 255, 0)
    # if (cv2.__version__[0] > 3):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # else:
    #     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) != 0:
        # draw in blue the contours that were foundeds
        cv2.drawContours(image, contours, -1, 255, 3)

        # find the biggest countour (c) by the area
        c = max(contours, key = cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)

        # draw the biggest contour (c) in green
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    # show the images
    cv2.imshow("Result", image)

def circle_detect_values(image):
    img = image.copy()
    gray_blurred = cv2.medianBlur(image, 5)
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 30, param1=130, param2=15, minRadius=13, maxRadius=27)
    
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            
            cv2.circle(img, (a,b), r, (0, 255, 0), 2)
            cv2.circle(img, (a,b), 1, (255, 255, 0), 3)
    return img

def process_frame(image):
    ## Rotate/Crop Image
    image = rotate_crop(image)

    ## Brighten Image
#    image = adjust_gamma(image, 3.5)
    
    ## Find Circles
    with_pockets = table_detect_2(image)
    # with_circles = circle_detect(image)
    # with_pockets = pocket_detect(with_circles)
    return with_pockets

def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

## MAIN

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    circle_frame = process_frame(frame)
    # cv2.imshow('frame',circle_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

