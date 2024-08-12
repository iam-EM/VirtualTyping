import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import time

# Initialize the video capture
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the hand detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Define the keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

# Initialize the text that will be displayed as typed
finalText = ""

# Define a class for the keyboard buttons
class Button():
    def __init__(self, pos, text, size=[50, 50]):
        self.pos = pos
        self.size = size
        self.text = text

# Function to draw all buttons on the image
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 150, 100), cv2.FILLED)
        cv2.putText(img, button.text, (x + 15, y + 35),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    return img

# Create the list of button objects
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([55 * j + 50, 55 * i + 50], key))

# Add erase button below the keyboard
buttonList.append(Button([55 * 4 + 50, 55 * 3 + 50], "Erase", size=[100, 50]))

# Initialize variables for key press cooldown
last_key_press_time = 0
key_cooldown = 0.9  # 300ms cooldown between key presses

# Main loop
while True:
    # Capture each frame from the camera
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the image

    # Detect hands in the frame
    hands, img = detector.findHands(img, flipType=False)

    # Draw the keyboard on the image
    img = drawAll(img, buttonList)

    # If hands are detected
    if hands:
        for hand in hands:
            lmList = hand["lmList"]  # Landmark list
            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                # Check if index finger is over the button
                if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                    cv2.rectangle(img, button.pos, (x + w, y + h), (0, 200, 150), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 15, y + 35),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                    # Calculate the distance between index and middle finger
                    l, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2])

                    # Check if click is detected (distance between fingers is small)
                    current_time = time.time()
                    if l < 30 and current_time - last_key_press_time > key_cooldown:
                        cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 200), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 15, y + 35),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

                        if button.text == "Erase":
                            finalText = finalText[:-1]
                        else:
                            finalText += button.text
                        last_key_press_time = current_time

    # Draw the text box below the keyboard
    cv2.rectangle(img, (50, 400), (1230, 500), (0, 150, 100), cv2.FILLED)
    cv2.putText(img, finalText[-30:], (60, 475),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    # Display the image
    cv2.imshow("Virtual Keyboard", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
