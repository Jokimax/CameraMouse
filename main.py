import cv2
import mouse
import time
import asyncio
import argparse

# Start recording and load cascades
cam = cv2.VideoCapture(0)
_mouse = cv2.CascadeClassifier('Cascades\closed_frontal_palm.xml')
leftClick = cv2.CascadeClassifier('Cascades\\fist.xml')
rightClick = cv2.CascadeClassifier('Cascades\palm.xml')

# Parse user inputed arguments.
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--sensitivity', default=100, type=int, help='Mouse sensitivity. Default is 100')
parser.add_argument('-d', '--debug', default='f', help='See what the camera is detecting')
args = parser.parse_args()
sensitivity = args.sensitivity
debugMode = False
if(args.debug.startswith('t') or args.debug.startswith('T')):
    debugMode = True

# A check to whether we stopped moving the mouse will be used later if debug mode is enabled
finished = False

# Moves the mouse from it's current position towards where your hand is
def moveMouse(data, frame, drag = False):
    # Find out how much we need to move
    cameraHeight = frame.shape[0]
    cameraWidth = frame.shape[1]
    halfHeight = cameraHeight * 0.5
    halfWidth = cameraWidth * 0.5
    y = (data[1] - halfHeight)/halfHeight * sensitivity
    x = (data[0] - halfWidth)/halfWidth * sensitivity


    # If we need to left click while moving the mouse drag it
    if drag:
        mouse.drag(x, y, absolute=False, duration=0.05)
        return
    # Otherwise just move it.
    mouse.move(x, y, absolute=False, duration=0.05)

# Handles the logic for mouse inputs
async def cameraMouse(frame):
    global finished
    # Check whether we are making the hand gesture for moving the mouse
    found = _mouse.detectMultiScale(frame, minNeighbors=5)
    if(len(found) > 0):
        moveMouse(found[0], frame)
        finished = True
        return
    
    # Do the same for a right click
    found = rightClick.detectMultiScale(frame, minNeighbors=5)
    if(len(found) > 0):
        mouse.click(button='right')
        finished = True
        return
    
    # Aswell as an left click
    found = leftClick.detectMultiScale(frame, minNeighbors=5)
    if(len(found) > 0):
        mouse.click(button='left')
        # Wait and check whether we are still holding up the hand
        time.sleep(0.1)
        ret, frame = cam.read()
        found = leftClick.detectMultiScale(frame, minNeighbors=5)
        # If so keep drag the mouse until we put the hand down.
        while(found and ret):
            moveMouse(found[0], frame, drag = True)
            ret, frame = cam.read()
            found = leftClick.detectMultiScale(frame, minNeighbors=5)
        finished = True
        return
    finished = True

# Draws rectangles around detected objects
def highlightDetections(detections, frame):
    for (x, y, width, height) in detections:
        cv2.rectangle(frame, (x, y), 
                    (x + height, y + width), 
                    (0, 255, 0), 5)
    return frame

# Enabled with debug mode shows the camera and shows all detected objects
async def debugCamera(frame):
    global finished
    # Detect all possible input types
    found = list(_mouse.detectMultiScale(frame, minNeighbors=5))
    found.extend(list(leftClick.detectMultiScale(frame, minNeighbors=5)))
    found.extend(list(rightClick.detectMultiScale(frame, minNeighbors=5)))
    # Highlight all the inputs
    if(len(found) > 0):
        frame = highlightDetections(found, frame)
    cv2.imshow('frame', frame)

    # If the cameraMouse function is not finished keep doing so
    while not finished:
        ret, frame = cam.read()
        while not ret:
            time.sleep(0.05)
            ret, frame = cam.read()
        found = list(_mouse.detectMultiScale(frame, minNeighbors=5))
        found.extend(list(leftClick.detectMultiScale(frame, minNeighbors=5)))
        found.extend(list(rightClick.detectMultiScale(frame, minNeighbors=5)))
        if(len(found) > 0):
            frame = highlightDetections(found, frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) == ord('q'):
            break

# Our updater function
async def tick():
    global finished
    ret, frame = cam.read()
    # If the camera is disabled for whatever reason sleep
    if not ret:
        time.sleep(0.1)
        return
    
    # Start our needed functions the cameraMouse and optionally our debugger
    tasks = [cameraMouse(frame)]
    if debugMode:
        finished = False
        tasks.append(debugCamera(frame))
    await asyncio.gather(*tasks)
    cv2.waitKey(1)

if __name__ == '__main__':
    while True:
        asyncio.run(tick())