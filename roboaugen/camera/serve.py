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

import json
import os
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


template_dir = os.path.abspath('roboaugen/camera/server/templates')
static_dir = os.path.abspath('roboaugen/camera/server/static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.config['SECRET_KEY'] = 'secret!'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['TEMPLATES_AUTO_RELOAD'] = True

socketio = SocketIO(app, async_mode=None)

global end_to_end_transformation_estimator
end_to_end_transformation_estimator = None
if end_to_end_transformation_estimator is None:
    end_to_end_transformation_estimator = EndToEndTransformationEstimator()

global initial_state
initial_state = None

global final_state
final_state = None

global image_initial_raw
image_initial_raw = None

global image_final_raw
image_final_raw = None

global capture_image_task_queue
capture_image_task_queue = None
if capture_image_task_queue is None:
    capture_image_task_queue = Queue()

global capture_image_result_queue
capture_image_result_queue = None
if capture_image_result_queue is None:
    capture_image_result_queue = Queue()

def record_camera():
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, image = video_capture.read()
        image_capture_requested = False
        while not capture_image_task_queue.empty():
            image_capture_requested = True
            task = capture_image_task_queue.get()
        if image_capture_requested:
            capture_image_result_queue.put(image)


global video_recording_thread
video_recording_thread = None
if video_recording_thread is None:
    video_recording_thread = Thread(target=record_camera)
    video_recording_thread.daemon = True
    video_recording_thread.start()


global sample_index
sample_index = 1




class SolutionType():
    SOLUTION_FOUND = 'SOLUTION_FOUND'
    NO_SOLUTION_FOUND = 'NO_SOLUTION_FOUND'
    NOT_ENOUGH_STATES = 'NOT_ENOUGH_STATES'
class SolutionResponse():

    def __init__(self, solution_type, solution=None):
        self.solution_type = solution_type
        self.solution = solution

    def to_json_string(self):
        if self.solution is not None:
            return self.solution.to_json_string()
        else:
            return "{}"


#def generated_media():

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')

@app.route('/message', methods=['POST'])
def message():
    global initial_state, final_state, end_to_end_transformation_estimator, image_initial_raw, image_final_raw, sample_index
    state = request.get_json()
    state = RobotState.from_dictionary(state)
    print(state.__dict__)
    final_state = initial_state
    initial_state = state
    image_final_raw = image_initial_raw
    capture_image_task_queue.put('RECORD_CAMERA')
    result_ready = not capture_image_result_queue.empty()
    while not result_ready:
        result_ready = not capture_image_result_queue.empty()
    image_initial_raw = capture_image_result_queue.get()
    if final_state is not None and initial_state is not None:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
        config = Config()
        height, width = image_final_raw.shape[0], image_final_raw.shape[1]

        image_initial = np.copy(image_initial_raw)
        image_final = np.copy(image_final_raw)
        threshold = 0.1
        epipolar_threshold = 10
        end_to_end_solution = end_to_end_transformation_estimator.compute_transformation(initial_state,
            final_state, image_initial, image_final, threshold)

        visual_targets_initial, visual_predictions_initial, visual_suports_initial = end_to_end_solution.heatmap_images[0]
        visual_targets_final, visual_predictions_final, visual_suports_final = end_to_end_solution.heatmap_images[1]


        data_log_dir = f'data/log/{sample_index}'
        if not os.path.exists(data_log_dir):
            os.makedirs(data_log_dir)
        cv2.imwrite(f'{data_log_dir}/image_initial.jpg', image_initial)
        cv2.imwrite(f'{data_log_dir}/image_final.jpg', image_final)
        cv2.imwrite(f'{data_log_dir}/visual_predictions_initial.jpg', visual_predictions_initial)
        cv2.imwrite(f'{data_log_dir}/visual_predictions_final.jpg', visual_predictions_final)
        if end_to_end_solution.image_initial_procrustes is not None:
            cv2.imwrite(f'{data_log_dir}/image_initial_procrustes.jpg', end_to_end_solution.image_initial_procrustes )
        if end_to_end_solution.image_final_procrustes is not None:
            cv2.imwrite(f'{data_log_dir}/image_final_procrustes.jpg', end_to_end_solution.image_final_procrustes )
        if end_to_end_solution.image_initial_grouped is not None:
            cv2.imwrite(f'{data_log_dir}/image_initial_grouped.jpg', end_to_end_solution.image_initial_grouped )
        if end_to_end_solution.image_final_grouped is not None:
            cv2.imwrite(f'{data_log_dir}/image_final_grouped.jpg', end_to_end_solution.image_final_grouped )
        if end_to_end_solution.image_initial_ungrouped is not None:
            cv2.imwrite(f'{data_log_dir}/image_initial_ungrouped.jpg', end_to_end_solution.image_initial_ungrouped )
        if end_to_end_solution.image_final_ungrouped is not None:
            cv2.imwrite(f'{data_log_dir}/image_final_ungrouped.jpg', end_to_end_solution.image_final_ungrouped )
        with open(f'{data_log_dir}/initial_state.json', 'w') as outfile:
            json.dump(initial_state.__dict__, outfile)
        with open(f'{data_log_dir}/final_state.json', 'w') as outfile:
            json.dump(final_state.__dict__, outfile)

        sample_index = sample_index + 1


        visual_predictions_initial = base64.b64encode(cv2.imencode('.jpg', visual_predictions_initial)[1]).decode("utf-8")
        visual_predictions_final = base64.b64encode(cv2.imencode('.jpg', visual_predictions_final)[1]).decode("utf-8")
        image_initial_procrustes = base64.b64encode(cv2.imencode('.jpg', end_to_end_solution.image_initial_procrustes)[1]).decode("utf-8") \
            if end_to_end_solution.image_initial_procrustes is not None else ""
        image_final_procrustes = base64.b64encode(cv2.imencode('.jpg', end_to_end_solution.image_final_procrustes)[1]).decode("utf-8") \
            if end_to_end_solution.image_final_procrustes is not None else ""
        image_initial_grouped = base64.b64encode(cv2.imencode('.jpg', end_to_end_solution.image_initial_grouped)[1]).decode("utf-8") \
            if end_to_end_solution.image_initial_grouped is not None else ""
        image_final_grouped = base64.b64encode(cv2.imencode('.jpg', end_to_end_solution.image_final_grouped)[1]).decode("utf-8") \
            if end_to_end_solution.image_final_grouped is not None else ""




        data = {
            'visual_predictions_initial': visual_predictions_initial,
            'visual_predictions_final': visual_predictions_final,
            'image_initial_procrustes': image_initial_procrustes,
            'image_final_procrustes': image_final_procrustes,
            'image_initial_grouped': image_initial_grouped,
            'image_final_grouped': image_final_grouped
        }
        socketio.emit('camera_updated', data)

        if end_to_end_solution.solution is not None:
            return SolutionResponse(SolutionType.NO_SOLUTION_FOUND, end_to_end_solution.solution).to_json_string(), 200
        else:
            return SolutionResponse(SolutionType.NO_SOLUTION_FOUND).to_json_string(), 404
    #task_queue.put(message)
    return SolutionResponse(SolutionType.NOT_ENOUGH_STATES).to_json_string(), 404

@app.route('/')
def index():
    return render_template('index.html', async_mode=socketio.async_mode)

if __name__ == '__main__':
    socketio = SocketIO(app, async_mode=None)
    # NOTE: debug and reloader can not be used as otherwise the file is loaded twice causing loading the controller also twice
    socketio.run(app, host='0.0.0.0', debug=True, use_reloader=False, port=7000)
