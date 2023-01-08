import numpy as np
import cv2
import os
import boto3
import time
from imutils.video import FileVideoStream
import requests
import psycopg2


# Upload file to S3 bucket
def uploadFile(filename, bucket):
    s3 = boto3.resource('s3')
    data = open(filename, 'rb')
    s3.Bucket(bucket).put_object(Key=filename, Body=data)


# Determining if a calculated speed is valid, not something much different than previous data points
def validspeed(speedlist, speed):
    speedlist = sorted(speedlist)
    medianspeed = speedlist[len(speedlist) // 2]
    if speed < medianspeed - 20 or speed > medianspeed + 20:
        return False
    return True


# Determine which direction car is going in which determines pixels/inch speed
def checkDirection(x):
    if x > 640:
        return 0.863, "left"
    else:
        return 0.69, "right"


# Use some image recognition to determine color of car
def carColor():
    return "unknown"


def getWeather():
    lat = 42.509255
    lon = -71.084968
    complete_url = "https://api.openweathermap.org/data/3.0/onecall?lat={}&lon={}&units=imperial&appid={}".format(lat, lon, os.environ.get("openweathermap-apikey"))
    response = requests.get(complete_url)
    return response.json()


def carMake():
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

                prevx = None
                speedvals = []
                videofps = 30
                inchesperpixel = 0
                directionHeaded = None
                color = carColor()
                car_make = carMake()
                videoTime = video[11:18]

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
                            if prevx is None:
                                inchesperpixel, directionHeaded = checkDirection(x)
                                prevx = x
                            else:
                                inches = abs(x - prevx) * inchesperpixel  # inches in 1/30 of a second
                                speedinmph = inches * (videofps / 17.6)  # should be mph as in/s -> mph is 1/17.6
                                if len(speedvals) <= 10 and speedinmph > 10:
                                    speedvals.append(speedinmph)
                                elif speedinmph >= 10 and validspeed(speedvals, speedinmph):
                                    speedvals.append(speedinmph)
                                prevx = x
                            valid_cntrs.append(cntr)

                    cv2.waitKey(1)

                uploadFile("recordings/{}".format(video), "speedometer-1")

                finalMedianSpeed = sorted(speedvals)[len(speedvals) // 2]
                weather = getWeather()["current"]
                temperature = weather["temp"]
                visibility = weather["visibility"]
                clouds = weather["clouds"]
                wind_speed = weather["wind_speed"]
                weather_description = weather["weather"][0]["description"]


                connection = psycopg2.connect(
                    host='speedometer-third-try.coelkjdianuh.us-east-2.rds.amazonaws.com',
                    port=5432,
                    user='postgres',
                    password=os.environ.get('pg-db-password'),
                    database=os.environ.get('pg-db-name')
                )
                cursor = connection.cursor()
                cursor.execute("""INSERT INTO speedometer 
                VALUES (%(recording_name)s, %(speed)s, %(direction)s, %(color)s, %(car_make)s, %(time)s, 
                %(temperature)s, %(visibility)s, %(clouds)s, %(wind_speed)s, %(weather_description)s); """,
                {'recording_name': video, 'speed': finalMedianSpeed, 'direction': directionHeaded, 'color': color,
                 'car_make': car_make, 'time': videoTime, 'temperature': temperature, 'visibility': visibility,
                 'clouds': clouds, 'wind_speed': wind_speed, 'weather_description': weather_description})
                connection.commit()
                cursor.close()
                connection.close()

                os.remove("recordings/{}".format(video))

        time.sleep(300)


if __name__ == '__main__':
    main()
