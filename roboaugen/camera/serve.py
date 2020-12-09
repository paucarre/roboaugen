from flask import Flask, request, send_from_directory,  Response, render_template, jsonify
from flask import jsonify
from threading import Thread
from multiprocessing import Process, Queue
from flask_socketio import SocketIO, emit
import traceback
import requests
import cv2
import numpy as np
import logging

from robotcontroller.kinematics import RobotTopology, RobotState

from roboaugen.core.config import Config
from roboaugen.model.inference import Inferencer
from roboaugen.camera.triangularization import FundamentalMatrixGenerator, KeypointMatcher, EpipolarLineGenerator, Triangularizer
from roboaugen.camera.model import CameraModel
from roboaugen.camera.transformation_estimator import ProcrustesProblemSolver
from roboaugen.camera.triangularization import EndToEndTransformationSolution, EndToEndTransformationEstimator

import torch

def to_radians(degrees):
    return ( degrees * np.pi ) / 180.


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['TEMPLATES_AUTO_RELOAD'] = True
socketio = SocketIO(app, async_mode=None)

global task_queue
global inference_process
task_queue = None
inference_process = None
if task_queue is None:
    task_queue = Queue()
if inference_process is None:
    print("CREATING INFERENCE PROCESS")
    #robot_communication = get_robot_communication(True)
    #control_process = control(task_queue, robot_communication)

global inferencer
inferencer = None
if inferencer is None:
    inferencer = Inferencer(distort=False, keep_dimensions=True, use_cache=False, \
        mode='silco', max_background_objects=1, max_foreground_objects=1)

global supports
supports = None
if supports is None:
    supports_folder = 'test/images/'
    supports = inferencer.get_supports_from_folder(supports_folder)
    supports = torch.cat([supports] * 2 ).cuda()

global video_capture
video_capture  = None
if video_capture is None:
    video_capture = cv2.VideoCapture(1)



@app.route('/message', methods=['POST'])
def message():
    message = request.get_json()
    task_queue.put(message)
    return 'OK', 200

@app.route('/', methods=['GET'])
def index():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    config = Config()
    logging.info('START')
    ret, image_initial_raw = video_capture.read()
    ret, image_final_raw = video_capture.read()
    logging.info('IMAGES LOADED')
    height, width = image_final_raw.shape[0], image_final_raw.shape[1]
    camera_topology = RobotTopology(l1=142, l2=142, l3=80, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
    initial_state = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(70.), angle_3=to_radians(-20.))
    final_state   = RobotState(linear_1=5, angle_1=to_radians(0.), angle_2=to_radians(-90.), angle_3=to_radians(20.))
    end_to_end_transformation_estimator = EndToEndTransformationEstimator()
    solution = end_to_end_transformation_estimator.compute_transformation(initial_state, final_state, image_initial_raw, image_final_raw)
    print(solution)
    return 'OK', 200

if __name__ == '__main__':
    socketio = SocketIO(app, async_mode=None)
    # NOTE: debug and reloader can not be used as otherwise the file is loaded twice causing loading the controller also twice
    socketio.run(app, host='0.0.0.0', debug=True, use_reloader=False, port=7000)
