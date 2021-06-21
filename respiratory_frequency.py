import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys


if __name__ == '__main__':
    # Import Video
    file_path = sys.argv[1]

    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully
    if cap.isOpened() == False:
        print("Error opening video stream or file")

    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video {video_width}x{video_height}")
    face_cascade_name = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_name)
    eyes_cascade_name = 'haarcascade_eye.xml'
    eyes_cascade = cv2.CascadeClassifier(eyes_cascade_name)

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # TODO: Classificar
        faces = face_cascade.detectMultiScale(gray)
        eyes = eyes_cascade.detectMultiScale(gray)
        # TODO: Desenhar retangulo
        for xface, yface, wface, hface in faces:
            cv2.rectangle(frame, (xface, yface), (xface+wface, yface+hface), (255, 0, 0))
            faceROI = gray[yface:yface+hface,xface:xface+wface]
            #print(faceROI)
            eyes = eyes_cascade.detectMultiScale(faceROI)
            
            for xeyes, yeyes, weyes, heyes in eyes:
                cv2.rectangle(frame, (xeyes + xface, yeyes + yface), (xeyes+xface+weyes, yeyes+yface+heyes), (0, 0, 255))
                forehead = frame[xeyes +int(weyes *0.25):yface, int(weyes *0.5):int((yeyes - yface)*0.6)]
                print(forehead)
                #cv2.imshow('Testa', forehead)
        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
