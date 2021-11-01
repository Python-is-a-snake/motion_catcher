import cv2
first_frame = None


video = cv2.VideoCapture(0)
#main loop of program
while True:
    #getting video from camera
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlurr(gray, (21, 21), 0)


    if first_frame is None:
        first_frame = gray
        continue
    #frame with delta filter
    delta_frame = cv2.absdiff(first_frame, gray)
    #frame with thresh filter
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]

    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    (cnts,_) =cv2.findCountours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 1000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)


    #printing frames
    cv2.imshow("GRAY", gray)
    cv2.imshow("DELTA", delta_frame)
    cv2.imshow("THRESH", thresh_frame)
    cv2.imshow("COLOR", frame)
    key = cv2.waitKey(1)
    print(gray)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows
