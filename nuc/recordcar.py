import os

import numpy as np
import cv2
import datetime
import shutil
import time


def main():
    video = cv2.VideoCapture(0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prevframe = None
    empty_frame_counter = 6
    uploaded_file = False
    file_name = None
    out = None
    written_frames = 0
    t = time.process_time()
    seconds = 0
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

        empty_frame = True
        for cntr in contours:
            # Setting the minimum size of something to be a contour
            if cv2.contourArea(cntr) >= 1500:
                empty_frame = False
                if empty_frame_counter >= 5:
                    file_name = "unfinished_recordings/{}".format(datetime.datetime.now()).replace(' ', '-').replace(
                        '.', '_').replace(':', '-') + ".mp4v"
                    out = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'DIVX'), 30, (1280, 720))
                empty_frame_counter = 0

        cv2.waitKey(1)
        prevframe = frame
        if empty_frame:
            empty_frame_counter += 1

        if empty_frame_counter <= 5:
            out.write(frame)
            written_frames += 1
            elapsed_time = time.process_time() - t
            if elapsed_time - seconds > 1:
                print("Written frames: {}".format(written_frames))
                print(elapsed_time)
                seconds += 1
            uploaded_file = False
        else:
            if not uploaded_file and file_name is not None:
                out.release()
                shutil.copyfile(file_name, "recordings/{}".format(file_name.split("/")[1]))

        print("Empty frame counter: {}".format(empty_frame_counter))


if __name__ == '__main__':
    main()
