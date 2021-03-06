from keras.models import load_model
import cv2
import numpy as np
import math
from phue import Bridge
import time
from keras.models import load_model
import cv2
import numpy as np
import math
from phue import Bridge
import time
import copy
import smtplib
import pywhatkit as kit
from email.message import EmailMessage

def predict_rgb_image(img):
    result = gesture_names[model.predict_classes(img)[0]]
    print(result)
    return (result)

def predict_rgb_image_vgg(image):
    image = np.array(image, dtype='float32')
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learningRate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def SMS_alert(subject, message, to):
    msg = EmailMessage()
    msg.set_content(message)
    msg['subject'] = subject
    msg['to'] = to
   
    user = "garagealerter@gmail.com"
    msg['from'] = user
    password = "uqpflkcgtbangkui"

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(user, password)
    server.send_message(msg)

    server.quit()

# General Settings
prediction = ''
action = ''
score = 0
img_counter = 500

# Parameters
cap_region_x_begin = 0.6  # Start point/total width
cap_region_y_end = 0.5  # Start point/total width
threshold = 60  # Binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# Variables
bgCap = 0
garageOpened = False

gesture_names = {0: 'Fist',
                 1: 'L',
                 2: 'Okay',
                 3: 'Palm',
                 4: 'Peace'}

# Change load_model to wherever VGG_cross_validated.h5 file is located
model = load_model('/Register Hand Gestures/VGG_cross_validated.h5')

#Phue
bridge_ip = '192.168.1.83'
b = Bridge(bridge_ip)
on_command = {'transitiontime': 0, 'on': True, 'bri': 254}
off_command = {'transitiontime': 0, 'on': False, 'bri': 254}
l = b.lights
                
capture = cv2.VideoCapture(0)

while(capture.isOpened()):
    ret, frame = capture.read()
    frame = cv2.bilateralFilter(frame, 5, 50, 100)  # Smoothing filter
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    
    cv2.imshow('frame',frame)

    if bgCap == 1:
        img = remove_background(frame)
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # Clip the ROI
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)

        cv2.imshow('blur', blur)
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Add prediction and action text to thresholded image
        cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  
        
        # Draw the text
        cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
        cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255))
        cv2.imshow('ori', thresh)

        # Get the contours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1

        if length > 0:
            for i in range(length):  # Find the biggest contour (according to area)
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

        cv2.normalize(frame, frame, 50, 255, cv2.NORM_MINMAX)
        cv2.imshow('output', drawing)
    
    k = cv2.waitKey(5) & 0xFF

    if k == 27:
        break
    elif k == ord('b'):  # Press 'b' to capture the background
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
        b.set_light(1, on_command)
        b.set_light(2, on_command)
        time.sleep(2)
        bgCap = 1
        print('Background captured')
    elif k == ord('r'):
        time.sleep(1)
        bgModel = None
        bgCap = 0
        print('Reset bg')
    elif k == 32:
        cv2.imshow('original', frame)
        target = np.stack((thresh,) * 3, axis=-1)
        target = cv2.resize(target, (224, 224))
        target = target.reshape(1, 224, 224, 3)
        prediction, score = predict_rgb_image_vgg(target)

        # Functionalities
        if prediction == "Peace":
            try:
                action = "Lights on"
                for light in l:
                    light.on = True
                    light.brightness = 255
            except:
                pass

        # @txt.att.net changes depending on your phone carrier/provider. https://www.digitaltrends.com/mobile/how-to-send-a-text-from-your-email-account/ lists all the extensions
        elif prediction == "Okay":
            if garageOpened == False:
                SMS_alert("Garage Alert", "Your garage has been opened", "8582264394@txt.att.net")
                garageOpened = True
            elif garageOpened == True:
                SMS_alert("Garage Alert", "Your garage has been closed", "8582264394@txt.att.net")
                garageOpened = False

        elif prediction == "Palm":
            kit.playonyt("lofi hip hop radio - beats to relax/study to")

cv2.destroyAllWindows()
capture.release()
