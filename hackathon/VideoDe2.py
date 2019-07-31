from imageai.Detection import VideoObjectDetection
import os
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
execution_path = os.getcwd()


cap = cv2.VideoCapture(1)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('video.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:

         #写入帧
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()


detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

video_path = detector.detectObjectsFromVideo(input_file_path = os.path.join(execution_path,"video.avi"),
                                             output_file_path=os.path.join(execution_path, "camera_detected_video.avi"),
                                             frames_per_second=50, log_progress=True, minimum_percentage_probability=30)
print(video_path)
# above is  图像识别



