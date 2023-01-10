import numpy as np
import cv2
import os
import boto3
import time
from imutils.video import FileVideoStream
import requests
import psycopg2
import configparser


# Upload file to S3 bucket
def upload_file(filename, bucket):
    s3 = boto3.resource('s3')
    data = open(filename, 'rb')
    s3.Bucket(bucket).put_object(Key=filename, Body=data)


# Determining if a calculated speed is valid, not something much different than previous data points
def valid_speed(speed_list, speed):
    speed_list = sorted(speed_list)
    median_speed = speed_list[len(speed_list) // 2]
    if speed < median_speed - 20 or speed > median_speed + 20:
        return False
    return True


# Determine which direction car is going in which determines pixels/inch speed
def check_direction(x):
    if x > 640:
        return 0.863, "left"
    else:
        return 0.69, "right"


# Use some image recognition to determine color of car
def car_color():
    return "unknown"


def get_weather():
    lat = 42.509255
    lon = -71.084968
    complete_url = \
        "https://api.openweathermap.org/data/3.0/onecall?lat={}&lon={}&units=imperial&appid={}". \
            format(lat, lon, config['DEFAULT']['OWMAPIKEY'])
    response = requests.get(complete_url)
    return response.json()


def car_make():
    return "unknown"


def main():
    while True:
        if os.listdir("recordings"):
            for video in os.listdir("recordings"):
                print(video)
                # start the file video stream thread and allow the buffer to start to fill
                fvs = FileVideoStream("recordings/{}".format(video)).start()

                # kernel for image dilation
                kernel = np.ones((4, 4), np.uint8)

                # Gets a previous frame before the loop
                frame = fvs.read()

                prev_x = None
                speed_vals = []
                video_fps = 30
                inches_per_pixel = 0
                direction_headed = None
                color = car_color()
                make_of_car = car_make()
                video_time = video[11:19].replace("-", ":")

                while fvs.running():
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

                    # Use "close" morphological operation to close the gaps between contours
                    # https://stackoverflow.com/questions/18339988/implementing-imcloseim-se-in-opencv
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51)))

                    # image dilation
                    dilated = cv2.dilate(thresh, kernel, iterations=1)

                    # find contours
                    contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    # contours
                    valid_cntrs = []
                    for cntr in contours:
                        x, y, w, h = cv2.boundingRect(cntr)
                        # Setting the minimum size of something to be a contour
                        if cv2.contourArea(cntr) >= 1500:
                            if prev_x is None:
                                inches_per_pixel, direction_headed = check_direction(x)
                                prev_x = x
                            else:
                                inches = abs(x - prev_x) * inches_per_pixel  # inches in 1/30 of a second
                                speedinmph = inches * (video_fps / 17.6)  # should be mph as in/s -> mph is 1/17.6
                                if len(speed_vals) <= 10 and speedinmph > 10:
                                    speed_vals.append(speedinmph)
                                elif speedinmph >= 10 and valid_speed(speed_vals, speedinmph):
                                    speed_vals.append(speedinmph)
                                prev_x = x
                            valid_cntrs.append(cntr)

                    cv2.waitKey(1)

                # If there is a bugged video that doesn't have a car moving, or a car moving really slowly then
                # delete the video and skip doing any uploading
                if len(speed_vals) == 0:
                    os.remove("recordings/{}".format(video))
                    continue
                upload_file("recordings/{}".format(video), "speedometer-1")

                finalMedianSpeed = sorted(speed_vals)[len(speed_vals) // 2]
                weather = get_weather()["current"]
                temperature = weather["temp"]
                visibility = weather["visibility"]
                clouds = weather["clouds"]
                wind_speed = weather["wind_speed"]
                weather_description = weather["weather"][0]["description"]

                connection = psycopg2.connect(
                    host='speedometer-third-try.coelkjdianuh.us-east-2.rds.amazonaws.com',
                    port=5432,
                    user='postgres',
                    password=str(config['DEFAULT']['PGDBPASSWORD']),
                    database=str(config['DEFAULT']['PGDBNAME'])
                )
                cursor = connection.cursor()
                cursor.execute("""INSERT INTO speedometer 
                VALUES (%(recording_name)s, %(speed)s, %(direction)s, %(color)s, %(car_make)s, %(time)s, 
                %(temperature)s, %(visibility)s, %(clouds)s, %(wind_speed)s, %(weather_description)s); """,
                               {'recording_name': video, 'speed': finalMedianSpeed, 'direction': direction_headed,
                                'color': color,
                                'car_make': make_of_car, 'time': video_time, 'temperature': temperature,
                                'visibility': visibility,
                                'clouds': clouds, 'wind_speed': wind_speed, 'weather_description': weather_description})
                connection.commit()
                cursor.close()
                connection.close()

                os.remove("unfinished_recordings/{}".format(video))
                os.remove("recordings/{}".format(video))

        # time.sleep(300)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('speedometer_secrets.ini')
    main()
