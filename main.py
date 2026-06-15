import cv2
from ultralytics import YOLO

# Load the newly downloaded medium model
model = YOLO('yolov8m.pt') 


cap = cv2.VideoCapture(r'D:\a\object tracking\highway_vehicles.mp4')
object_detector = cv2.createBackgroundSubtractorMOG2()
while True:
    ret, frame = cap.read()
    # height, width, _ = frame.shape
    # print(height, width)
    # Extract region of interest
    roi = frame[400:1080, 0:700]


    mask = object_detector.apply(roi)

    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
        # calculate area and remove small elements
        # area = cv2.contourArea(cnt)
        # if area > 100:
            # cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
            # x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(roi, (x,y), (x+w, y+h), (0,255,0), 3)


    results = model.track(
        source=roi,
        show=False, 
        persist=True,               
        tracker='botsort.yaml',    
        conf=0.40,                 
        iou=0.5,                   
        classes=[2,7],            # 2 = car, 7 = truck 
        single_cls=True,           # Stops label flickering
        verbose=False              # Cleans up your command terminal outputs
    )

    if results and len(results) > 0:
        annotated_roi = results[0].plot()
    else:
        annotated_roi = roi

    cv2.imshow('Tracking & Contours (ROI)', annotated_roi)
    # cv2.imshow('frame', frame)
    # cv2.imshow('mask', mask)
    key = cv2.waitKey(30)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break



    

cap.release()
cv2.destroyAllWindows()


