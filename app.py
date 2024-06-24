import os
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox #box around the objects
from gtts import gTTS
from playsound import playsound

# Define the paths
# yolo_dir = os.path.expanduser('~/cvlib/yolo')
# weights_path = os.path.join(yolo_dir, 'yolov3.weights')
# config_path = os.path.join(yolo_dir, 'yolov3.cfg')
# labels_path = os.path.join(yolo_dir, 'coco.names')

#activation video
video = cv2.VideoCapture(0)         #index is linked to the quality of the camera ("mp4")

#test video opening, error is not
if not video.isOpened():
    print("Error: Could not open video stream.")

#loading images to our data
output_dir = 'output images'
os.makedirs(output_dir, exist_ok=True)

img_counter = 0


while True:
    #use video capture and unpack each frame into variable frame
    ret, frame = video.read()

    #check if frame was captured:
    if not ret:
        print("Error: Failed to capture frame.")
        break
    cv2.imshow('Webcam', frame)

    k = cv2.waitKey(1)
    #if we hit esc to quit:
    if k%256 == 27:
        print("Escape hit, closing...")
        break
    #saving images when press 's':
    elif k%256 == ord("s"):
        img_name = os.path.join(output_dir, "opencv_frame{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter +=1

video.release()
cv2.destroyAllWindows()


# while True:
#     #use video capture and unpack each frame into variable frame
#     ret, frame = video.read()

#     #drawing box around object + label 
#     bbox,label, conf = cv.detect_common_objects(frame,                  #conf returns some decimals, identifying the object
#                                                 confidence=0.2,
#                                                 model='yolov3',
#                                                 enable_gpu=False,
#                                                 )             
#     output_image = draw_bbox(frame, bbox, label, conf)

#     #show the user how it's going to look like
#     cv2.imshow("Object Detection", output_image)

#     if cv2.waitKey(1) & 0xFF == ord(" "):                    #if users clicks on some key, break out of the loop
#         break

# #ensures that the video capture object is released and all OpenCV windows are closed when the loop ends.
# video.release()
# cv2.destroyAllWindows()



