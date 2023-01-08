import numpy as np
import cv2
import datetime


def main():
    ########## Taken from realdetection.py ###########
    video = cv2.VideoCapture(0)

    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ####################################################

    prevframe = None
    emptyFrameCounter = 6
    uploadedFile = False
    fileName = None
    out = None
    while True:
        check, frame = video.read()
        if prevframe is None:
            prevframe = frame

        # kernel for image dilation
        kernel = np.ones((4, 4), np.uint8)

        # frame differencing
        graya = cv2.cvtColor(prevframe, cv2.COLOR_BGR2GRAY)
        grayb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_image = cv2.absdiff(grayb, graya)

        # image thresholding
        ret, thresh = cv2.threshold(diff_image, 25, 255, cv2.THRESH_BINARY)

        # Use "close" morphological operation to close the gaps between contours
        # https://stackoverflow.com/questions/18339988/implementing-imcloseim-se-in-opencv
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))

        # image dilation
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        # find contours
        contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        emptyframe = True
        for cntr in contours:
            # Setting the minimum size of something to be a contour
            if cv2.contourArea(cntr) >= 1500:
                emptyframe = False
                if emptyFrameCounter >= 5:
                    fileName = "recordings/{}".format(datetime.datetime.now()).replace(' ', '-').replace('.', '_').replace(':', '-') + ".mp4v"
                    out = cv2.VideoWriter(fileName, cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720))
                emptyFrameCounter = 0

        cv2.waitKey(1)
        prevframe = frame
        if emptyframe:
            emptyFrameCounter += 1

        if emptyFrameCounter <= 5:
            out.write(frame)
            uploadedFile = False
        else:
            if not uploadedFile and fileName is not None:
                out.release()

        print(emptyFrameCounter)


if __name__ == '__main__':
    main()
