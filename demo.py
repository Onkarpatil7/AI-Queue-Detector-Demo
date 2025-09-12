import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened(): #checks if camera is opened
    print("Error: Could not open webcam")
    exit()

while True: #infinte loop
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    cv2.imshow("Webcam Feed", frame)

    
    if cv2.waitKey(30) & ord('q'):
        break

cap.release() #releases the camera for other usage
cv2.destroyAllWindows() #


