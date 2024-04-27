#!/usr/bin/env python3
from explode import explode_frame
# from mirror import mirror_frame
# from swirl import swirl_filter
from butterflies import apply_face_filter
from heart import apply_heart_filter

from flask import Flask, render_template, Response, request
from flask_socketio import SocketIO, send, emit
import threading
import time

import cv2
import numpy as np
from app import main

app = Flask(__name__)
app.config['SECRET_KEY'] = "adudu1234"
socketio = SocketIO(app)
# hihi
camera = cv2.VideoCapture(0)

@app.route("/hello")
def hello(): 
    return "hello"

def modify_frames(frame):
    # print('hello')
    return frame


def generate_frames(filter_param):
    while True:
        frame_generator = main()
        for frame in frame_generator: 
            # if (filter_param == "explode"):
            #     flipped_frame = cv2.flip(explode_frame(frame), 1)
            #     print("hello")
            # elif (filter_param == "butterflies"):
            #     flipped_frame = cv2.flip(apply_face_filter(frame), 1)
            # elif (filter_param == "heart"):
            #     flipped_frame = cv2.flip(apply_heart_filter(frame), 1)
            # else:
            #     flipped_frame = cv2.flip(frame, 1)
            ret, buffer = cv2.imencode('.jpg', modify_frames(frame))
            buffer_frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer_frame + b'\r\n')
        # success, frame = main()
        # if not success:
        #     print("break")
        #     break
        # else:
        #     # if (filter_param == "explode"):
        #     #     flipped_frame = cv2.flip(explode_frame(frame), 1)
        #     # elif (filter_param == "butterflies"):
        #     #     flipped_frame = cv2.flip(apply_face_filter(frame), 1)
        #     # elif (filter_param == "heart"):
        #     #     flipped_frame = cv2.flip(apply_heart_filter(frame), 1)
        #     # else:
        #     #     flipped_frame = cv2.flip(frame, 1)


        #     ret, buffer = cv2.imencode('.jpg', modify_frames(frame))
        #     frame = buffer.tobytes()
        #     print(frame)

        # yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@socketio.on('connect')
def handle_connect():
    print('Client connected')


def send_message_to_nodejs(message):
    print('triggered')
    time.sleep(5) 
    socketio.emit('message_to_nodejs', message, namespace='/')


@app.route('/capture')
def capturePhoto():
    threading.Thread(target=send_message_to_nodejs, args=("Yooo wtf",)).start()
    return "hello"


@app.route('/video')
def video():
    filter_param = request.args.get('filter')
    print("testing1")
    return Response(generate_frames(filter_param), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    socketio.run(app, debug=True)
    # app.run(debug=True)