from flask import Flask, request, send_from_directory,  Response, render_template, jsonify
from flask import jsonify
from threading import Thread
from multiprocessing import Process, Queue
from flask_socketio import SocketIO, emit
import traceback
import requests

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

@app.route('/message', methods=['POST'])
def index():
    message = request.get_json()
    task_queue.put(message)
    return 'OK', 200

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', debug=True, port=7000)
