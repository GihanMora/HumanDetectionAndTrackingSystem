from flask import Flask, render_template, Response,request, jsonify

from core import VideoCamera
import cv2
import pymysql
import time
from Data_Access import testdb
db = pymysql.connect("localhost", "root", "", "dynamic")
from variables import static_variables

app = Flask(__name__)

class static_tracker:
    trackers_list=[]
    tracker_queues_list = []
    fname=""
    ttt = []
    restricted = []




@app.route('/background_process')
def background_process():


    #points = get_points.run(static_variables.frame)
    #static_variables.restricted.append(points[0])
    try:
        return jsonify(result='Done.')
    except Exception as e:
        return str(e)












@app.route('/')
def index():
    static_variables.fname = ""
    static_variables.Frame_count = 0
    static_variables.trackers_list = []
    static_variables.tracker_queues_list = []
    static_variables.restricted = []
    static_variables.lost_trackers = []
    static_variables.lost_detections = []
    static_variables.entrance_information = []
    static_variables.op = True
    static_variables.info_set=[]
    return render_template('Home.html')
@app.route('/video',methods=['GET','POST'])
def open_video():

    if request.method == 'POST':
        f = request.files['file']
        static_tracker.fname="Sample_Videos/"+f.filename


    return render_template('video.html')
testss="blazzzzzzzz"


@app.route('/dynamic')
def dyna():

    return render_template('index11.php')

@app.route('/camera',methods=['GET','POST'])
def open_cam():

    return render_template('camera.html')



def gen(camtest):
    while True:
        try:
            frame = camtest.get_frame()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


        except :
                print (Exception)
                cv2.destroyAllWindows()
                break


@app.route('/database')
def viewdata():
    results = testdb.getdata()
    return render_template('index.html', results=results)


@app.route('/video_feed')
def video_feed():
    static_variables.fname = ""
    static_variables.Frame_count = 0
    static_variables.trackers_list = []
    static_variables.tracker_queues_list = []
    static_variables.restricted = []
    static_variables.lost_trackers = []
    static_variables.lost_detections = []
    static_variables.op = True
    static_variables.entrance_information = []
    static_variables.info_set = []
    static_variables.camera_input = False
    return Response(gen(VideoCamera(static_tracker.fname)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam_feed')
def cam_feed():
    static_variables.fname = ""
    static_variables.Frame_count = 0
    static_variables.trackers_list = []
    static_variables.tracker_queues_list = []
    static_variables.restricted = []
    static_variables.lost_trackers = []
    static_variables.lost_detections = []
    static_variables.op = True
    static_variables.entrance_information = []
    static_variables.info_set = []
    static_variables.camera_input=True
    return Response(gen(VideoCamera(0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True)
