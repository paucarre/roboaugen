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
import base64
import struct
import json
from enum import Enum, auto

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

global end_to_end_transformation_estimator
end_to_end_transformation_estimator = None
if end_to_end_transformation_estimator is None:
    end_to_end_transformation_estimator = EndToEndTransformationEstimator()


global video_capture
video_capture  = None
if video_capture is None:
    video_capture = cv2.VideoCapture(1)

global initial_state
initial_state = None

global final_state
final_state = None

global image_initial_raw
image_initial_raw = None

global image_final_raw
image_final_raw = None


def decode_double(data):
    data = base64.b64decode(data)
    data = struct.unpack("d", data)[0]
    return data

def get_state_from_request(request_json):
    linear_1 = decode_double(request_json['linear_1'])
    angle_1 = decode_double(request_json['angle_1'])
    angle_2 = decode_double(request_json['angle_2'])
    angle_3 = decode_double(request_json['angle_3'])
    return RobotState(linear_1=linear_1, angle_1=angle_1, angle_2=angle_2, angle_3=angle_3)

class SolutionType():
    SOLUTION_FOUND = 'SOLUTION_FOUND'
    NO_SOLUTION_FOUND = 'NO_SOLUTION_FOUND'
    NOT_ENOUGH_STATES = 'NOT_ENOUGH_STATES'
class SolutionResponse():

    def __init__(self, solution_type, solution=None):
        self.solution_type = solution_type
        self.solution = solution

    def to_json_string(self):
        return json.dumps(self.__dict__, indent = 4)

@app.route('/message', methods=['POST'])
def message():
    global initial_state, final_state, video_capture, end_to_end_transformation_estimator, image_initial_raw, image_final_raw
    state = request.get_json()
    state = get_state_from_request(state)
    final_state = initial_state
    initial_state = state
    image_final_raw = image_initial_raw
    ret, image_initial_raw = video_capture.read()
    if final_state is not None and initial_state is not None:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
        config = Config()
        height, width = image_final_raw.shape[0], image_final_raw.shape[1]
        camera_topology = RobotTopology(l1=142, l2=142, l3=80, h1=50, angle_wide_1=180, angle_wide_2=180 + 90, angle_wide_3=180 + 90)
        solution = end_to_end_transformation_estimator.compute_transformation(initial_state, final_state, image_initial_raw, image_final_raw)
        if solution.solution is not None:
            return SolutionResponse(SolutionType.NO_SOLUTION_FOUND, solution.solution).to_json_string(), 200
        else:
            return SolutionResponse(SolutionType.NO_SOLUTION_FOUND).to_json_string(), 404
    #task_queue.put(message)
    return SolutionResponse(SolutionType.NOT_ENOUGH_STATES).to_json_string(), 404

@app.route('/', methods=['GET'])
def index():
    return 'OK', 200

if __name__ == '__main__':
    socketio = SocketIO(app, async_mode=None)
    # NOTE: debug and reloader can not be used as otherwise the file is loaded twice causing loading the controller also twice
    socketio.run(app, host='0.0.0.0', debug=True, use_reloader=False, port=7000)
