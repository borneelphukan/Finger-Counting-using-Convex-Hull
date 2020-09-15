import cv2
import numpy as np
import math

# ============== GLOBAL VARIABLES ===================
font = cv2.FONT_HERSHEY_SIMPLEX
text_color = (0, 0, 0)      # black
window_color = (0, 255, 0)  # green
hull_color = [0, 123, 255]  # orange
low_skintone = [0, 20, 70]
high_skintone = [20, 255, 255]

error_message = ['-', '?']

# ============== NUMERICAL FUNCTIONS ================
def cosineRule(side_a, side_b, side_c):
    return math.acos((side_b**2 + side_c**2 - side_a**2)/(2*side_b*side_c)) * 57

def distHullPoint(area, side_a):
    return (2 * area) / side_a

def calculateArea(side_a, side_b, side_c, semi_perimeter):
    return math.sqrt(semi_perimeter*(semi_perimeter-side_a)*(semi_perimeter-side_b)*(semi_perimeter-side_c))

# ============== I/O FUNCTIONS =====================
def printOutput(num_defects, hand_area, area_ratio):
    if num_defects == 1:
        if hand_area < 2000:
            cv2.putText(frame, error_message[0], (0, 50), font, 2, text_color, 3, cv2.LINE_AA)
        else:
            if area_ratio < 12:
                cv2.putText(frame, '0', (0, 50), font, 2, text_color, 3, cv2.LINE_AA)  
            else:
                cv2.putText(frame, '1', (0, 50), font, 2, text_color, 3, cv2.LINE_AA)       
    elif num_defects == 2:
        cv2.putText(frame, '2', (0, 50), font, 2, text_color, 3, cv2.LINE_AA)
    elif num_defects == 3:
          if area_ratio < 27:
                cv2.putText(frame, '3', (0, 50), font, 2, text_color, 3, cv2.LINE_AA)
    elif num_defects == 4:
        cv2.putText(frame, '4', (0, 50), font, 2, text_color, 3, cv2.LINE_AA)
    elif num_defects == 5:
        cv2.putText(frame, '5', (0, 50), font, 2, text_color, 3, cv2.LINE_AA)
    else:
        cv2.putText(frame, error_message[1], (10, 50), font, 2, text_color, 3, cv2.LINE_AA)

# ============== MAIN SECTION (USING OPENCV) ========
video = cv2.VideoCapture(0)

while(1):

    try:
        # Setting the canvas for work
        _, frame = video.read()
        frame = cv2.flip(frame, 1)
        kernel = np.ones((3, 3), np.uint8)

        # Setting the range of interest for operation
        roi = frame[25:400, 25:400]

        # Setting the Window Frame
        cv2.rectangle(frame, (25, 25), (400, 400), window_color, 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Setting the skin tone
        lower_bound = np.array(low_skintone, dtype=np.uint8)
        upper_bound = np.array(high_skintone, dtype=np.uint8)

        '''
        SEGMENTATION OF HAND
        - extract skin colur from image
        - extrapolating the hand to fill the dark spaces within
        - image is blurred using Gaussian Blur method
        '''
        segment = cv2.inRange(hsv, lower_bound, upper_bound)
        segment = cv2.dilate(segment, kernel, iterations = 1)
        segment = cv2.GaussianBlur(segment, (5, 5), 100)

        # find contours
        contours, hierarchy = cv2.findContours(segment, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # find contour of max area(hand)
        max_contour_area = max(contours, key=lambda x: cv2.contourArea(x))

        # contour approximation
        epsilon = 0.0005 * cv2.arcLength(max_contour_area, True)
        approx = cv2.approxPolyDP(max_contour_area, epsilon, True)

        # convex hull around hand
        convex_hull = cv2.convexHull(max_contour_area)

        # define area of hull and hand
        hull_area = cv2.contourArea(convex_hull)
        hand_area = cv2.contourArea(max_contour_area)

        # Percentage of area not covered by hand
        area_ratio = ((hull_area-hand_area)/hand_area)*100

        # find the defects in convex hull with respect to hand
        convex_hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, convex_hull)

        num_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])

            side_a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            side_b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            side_c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            semi_perimeter = (side_a+side_b+side_c)/2
            area = calculateArea(side_a, side_b, side_c, semi_perimeter)

            distance = distHullPoint(area, side_a)      # distance between hull points and hull

            angle = cosineRule(side_a, side_b, side_c)  # angle between fingers

            # hull points between fingers
            if angle <= 90 and distance > 30:
                num_defects = num_defects + 1
                cv2.circle(roi, far, 3, [255, 255, 0], -1)

            # draw lines around hand
            cv2.line(roi, start, end, hull_color, 2)

        num_defects = num_defects + 1

        # prints the output
        printOutput(num_defects, hand_area, area_ratio)

        cv2.imshow('Finger Counting (1-5)', frame)
            
    except:
        pass

    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
video.release()
