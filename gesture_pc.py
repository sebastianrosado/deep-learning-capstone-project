import copy
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import cv2
import time
from pynput.keyboard import Key, Controller

keyboard = Controller()

# General Settings
prediction = ''
action = ''
score = 0
img_counter = 500

# Turn on/off the ability to save images
save_images, selected_gesture = False, 'blank'


# Control media
music = True

# Assign names to prediction index numbers
gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

# load json and create model
json_file = open('saved_models/model_sil_final.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("saved_models/model_sil_weights_final.h5")
print("Loaded model from disk")
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# parameters
cap_region_x_begin = 0.5  # start point/total width # was 0.5
cap_region_y_end = 0.8  # start point/total width # was 0.8
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0  # bool, whether the background captured
triggerSwitch = False  # if true, keyboard simulator works

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

# Set up list for VGG loop
frames = []


# Camera
camera = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
camera.set(10, 200)

if not camera.isOpened():
    raise Exception("Could not open camera")

while camera.isOpened():
    ret, frame = camera.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
    frame = cv2.flip(frame, 1)  # flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

#     Run once background is captured
    if isBgCaptured == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Add prediction and action text to thresholded image
        # Draw the text
        cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))
        cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255))  # Draw the text
        cv2.imshow('Binary', thresh)
        
        # Get the contours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            for i in range(length):  # find the biggest contour (according to area)
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i

            res = contours[ci]
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

        cv2.imshow('Contours', drawing)
        
        # Prepare camera stream to input every 16th frame into our VGG transformation pipeline
        frames.append(thresh)
        # copies 1 channel BW image to all 3 RGB channels
        input = np.stack((frames,) * 3, axis=-1)
        
        # Feed every 16th frame into processing pipeline for our model
        if input.shape[0] == 16:
            frames = [] # Clear frames so count resets every 32 frames
            input = input[0]
            target = cv2.resize(input, (224, 224))
            target = target.reshape(1, 224, 224, 3)
            target = np.array(target, dtype='float32')
            target /= 255
            pred_array = model.predict(target)
            print(f'pred_array: {pred_array}')
            result = gesture_names[np.argmax(pred_array)]
            print(f'Result: {result}')
            print(max(pred_array[0]))
            confidence = max(pred_array[0])
            score = float("%0.2f" % (max(pred_array[0]) * 100))
            print(result)
            input = []
            
            # Bind and execute media hotkeys with gesture predictions
            if (result == 'Palm' and confidence >= .96):
                try:
                    action = "Volume down"

                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)

                # Turn off media actions if devices are not responding
                except ConnectionError:
                    music = False
                    pass

            elif (result == 'Fist' and confidence >= .96):
                try:
                    action = 'Play / Pause'

                    keyboard.press(Key.media_play_pause)

                except ConnectionError:
                    music = False
                    pass

            elif (result == 'L' and confidence >= .99):
                try:
                    action = 'Volume up'

                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)

                except ConnectionError:
                    music = False
                    pass

            # Designate Okay as 'No Gesture' because of excessively frequent false positive issues
            elif (result == 'Okay' and confidence >= .99):
                try:
                    action = 'No Gesture'

                except ConnectionError:
                    music = False
                    pass

            elif (result == 'Peace' and confidence >= .96):
                try:
                    action = 'Next Track'
                    keyboard.press(Key.media_next)

                except ConnectionError:
                    music = False
                    pass

            else:
                try:
                    action = 'No Gesture'


                except ConnectionError:
                    music = False
                    pass
            

    # Keyboard OP
    k = cv2.waitKey(10)
    if k == 27:  # press ESC to exit all windows at any time
        break
    elif k == ord('b'):  # press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        time.sleep(2)
        isBgCaptured = 1
        print('Background captured')

    elif k == ord('r'):  # press 'r' to reset the background
        time.sleep(1)
        bgModel = None
        triggerSwitch = False
        isBgCaptured = 0
        print('Reset background')
        
        if save_images:
            img_name = f"C:\\Users\\sebar\\Documents'\\Sebastian\\Thinkful\\deep-learning-capstone-project\\captured_images\\drawings\\drawing_{selected_gesture}_{img_counter}.jpg".format(
                img_counter)
            cv2.imwrite(img_name, drawing)
            print("{} written".format(img_name))

            img_name2 = f"C:\\Users\\sebar\\Documents\\Sebastian\\Thinkful\\deep-learning-capstone-project\\captured_images\\{selected_gesture}_{img_counter}.jpg".format(
                img_counter)
            cv2.imwrite(img_name2, thresh)
            print("{} written".format(img_name2))

            img_name3 = f"C:\\Users\\sebar\\Documents\\Sebastian\\Thinkful\\deep-learning-capstone-project\\captured_images\\masks\\mask_{selected_gesture}_{img_counter}.jpg".format(
                img_counter)
            cv2.imwrite(img_name3, img)
            print("{} written".format(img_name3))

            img_counter += 1

    elif k == ord('t'):

        print('Tracker turned on.')

        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()

        # Select Region of Interest (ROI)
        r = cv2.selectROI(frame)

        # Crop image
        imCrop = frame[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # setup initial location of window
        r, h, c, w = 250, 400, 400, 400
        track_window = (c, r, w, h)
        # set up the ROI for tracking
        roi = imCrop
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        
        # Setup the termination criteria, either 10 iteration or move by at least 1 pt
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        while (1):
            ret, frame = cap.read()
            if ret == True:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
                # apply meanshift to get the new location
                ret, track_window = cv2.CamShift(dst, track_window, term_crit)
                # Draw it on image
                pts = cv2.boxPoints(ret)
                pts = np.int0(pts)
                img2 = cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                cv2.imshow('img2', img2)
                k = cv2.waitKey(60) & 0xff
                if k == 27:  # if ESC key
                    break
                else:
                    cv2.imwrite(chr(k) + ".jpg", img2)
            else:
                break
        cv2.destroyAllWindows()
        cap.release()
