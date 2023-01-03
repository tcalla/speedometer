from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import time
import cv2
import os
import initialdistance


# def validspeed(speedlist, speed):
#     speedlist = sorted(speedlist)
#     medianspeed = speedlist[len(speedlist) // 2]
#     if abs(speed - medianspeed) > medianspeed // 2:
#         return False
#     return True


def shownormalwindow(carx, cary, frame, speed, font, fps, speedvals):
    cv2.namedWindow("regular_window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("regular_window", 960, 540)
    cv2.moveWindow("regular_window", 0, 0)

    frame_copy = frame.copy()
    cv2.putText(frame_copy, "Current speed: {:.2f}".format(speed), (carx, cary - 60), font, 1.5, (0, 0, 200), 5)

    fps.update()
    # stop the timer and display FPS information
    fps.stop()
    cv2.putText(frame_copy, "FPS: {:.2f}".format(fps.fps()), (900, 80), font, 1.5, (0, 0, 200), 5)
    cv2.putText(frame_copy, "Elapsed time: {:.2f} secs".format(fps.elapsed()), (55, 80), font, 1.5,
                (0, 0, 200), 5)

    if len(speedvals) > 10:
        cv2.putText(frame_copy, "Median speed: {:.2f}".format(speedvals[len(speedvals) // 2]),
                    (carx, cary), font, 1.5, (0, 0, 200), 5)
    cv2.imshow("regular_window", frame_copy)


def showotherwindows(frame, valid_cntrs, font, speed, speedvals, diff_image, thresh):
    # add contours to original frames
    dmy = frame.copy()
    cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)

    cv2.putText(dmy, "Vehicles detected: " + str(len(valid_cntrs)), (55, 800), font, 2, (0, 0, 200), 7)
    cv2.putText(dmy, "Speed: {:.2f} mph".format(speed), (800, 800), font, 2, (0, 0, 200), 7)
    if len(speedvals) >= 10:
        cv2.putText(dmy, "Median Speed: {:.2f} mph".format(speedvals[len(speedvals) // 2]), (55, 900), font, 2,
                    (0, 0, 200), 7)

    # Image difference window
    cv2.namedWindow("Diff_image_window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Diff_image_window", 960, 540)
    cv2.moveWindow("Diff_image_window", 960, 0)
    cv2.imshow("Diff_image_window", diff_image)

    # Thresh window
    cv2.namedWindow("thresh_window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("thresh_window", 960, 540)
    cv2.moveWindow("thresh_window", 0, 540)
    cv2.imshow("thresh_window", thresh)

    # Countour window
    cv2.namedWindow("contour_window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("contour_window", 960, 540)
    cv2.moveWindow("contour_window", 960, 540)
    cv2.imshow("contour_window", dmy)


def main():
    # try to feed in all benchmark videos
    for video in os.listdir("Benchmark-Captures"):
        print(video)
        # start the file video stream thread and allow the buffer to start to fill
        fvs = FileVideoStream("Benchmark-Captures/{}".format(video)).start()
        time.sleep(1.0)

        # kernel for image dilation
        kernel = np.ones((4, 4), np.uint8)

        # font style
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Gets a previous frame before the loop
        frame = fvs.read()

        prevx = None
        speedvals = []
        key = None
        framecount = 0
        # initialdist = initialdistance.InitialDistance()
        videofps = 30
        fps = FPS().start()

        while fvs.running():

            framecount += 1

            # Call out to get the user to give a distance. If the user doesn't unput in 5 seconds it forces them to
            # if key == ord("m") or ((framecount > (videofps * 5)) and initialdist.inchesperpixel is None):
            #     cv2.imwrite("./firstframe.ppm", frame)
            #     while initialdist.run is True:
            #         initialdist.main()
            #         cv2.waitKey(0)
            #     os.remove("./firstframe.ppm")

            prevframe = frame[:]
            frame = fvs.read()
            # Always 1+ empty frames at end of video (images queue) that cause crash
            if frame is None:
                continue

            # frame differencing
            graya = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
            grayb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff_image = cv2.absdiff(grayb, graya)

            # image thresholding
            ret, thresh = cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)

            # image dilation
            dilated = cv2.dilate(thresh, kernel, iterations=1)

            # find contours
            contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

            # countours
            # Approx 0.61222 inches / pixel (for DJI 270 only)
            # For benchmarks, approx. 0.65 inches / pixel (1280 x 720p downscaled video from iPhone 13 pro)
            inchesperpixel = 0.65
            valid_cntrs = []
            cary = 0
            carx = 0
            speedinmph = 0
            for cntr in contours:
                x, y, w, h = cv2.boundingRect(cntr)
                # Setting the minimum size of something to be a contour
                # TODO: Take the max size contour only so only a single speed added
                if cv2.contourArea(cntr) >= 2000:
                    if prevx is None:
                        prevx = x
                    else:
                        # TODO: Give a margin for what the speed is? Like a range based on margin of error?
                        # inches = abs(x - prevx) * initialdist.inchesperpixel  # inches in 1/30 of a second
                        inches = abs(x - prevx) * inchesperpixel  # inches in 1/30 of a second
                        speedinmph = inches * (videofps / 17.6)  # should be mph as in/s -> mph is 1/17.6
                        # if len(speedvals) < 10 or validspeed(speedvals, speed) is True:
                        if speedinmph != 0:
                            speedvals.append(speedinmph)
                        prevx = x
                    valid_cntrs.append(cntr)
                    cary = y
                    carx = x

            # If there are no contours then continue processing and reset variables until there is something interesting
            # Also displays the first 5 seconds of video for the user to give initial measurement
            # if len(valid_cntrs) == 0 and (framecount > (videofps * 5) or initialdist.run is False):
            if len(valid_cntrs) == 0:
                # speedvals = []
                cv2.destroyWindow("Diff_image_window")
                cv2.destroyWindow("thresh_window")
                cv2.destroyWindow("contour_window")
                # Regular image window
                shownormalwindow(carx, cary, frame, speedinmph, font, fps, speedvals)
                # key = cv2.waitKey(1)
                cv2.waitKey(1)
                continue

            # Other windows
            showotherwindows(frame, valid_cntrs, font, speedinmph, speedvals, diff_image, thresh)

            # Regular image window
            # if initialdist.inchesperpixel is not None:
            shownormalwindow(carx, cary, frame, speedinmph, font, fps, speedvals)
            # else:
            #     shownormalwindow(carx, cary, frame, speed, font, None, speedvals)

            # 1ms delay which makes the windows appear as if it is video with pictures
            # key = cv2.waitKey(1)
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
