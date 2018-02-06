from flask import Flask, render_template, Response,request,redirect,jsonify
from camera import VideoCamera


app = Flask(__name__)

class static_tracker:
    tracker=[]
@app.route('/')
def index():
    return render_template('Home.html')
@app.route('/video',methods=['GET','POST'])
def open_video():

    return render_template('video.html')
testss="blazzzzzzzz"


@app.route('/test',methods=['GET','POST'])
def test():

    return render_template('test.html',name=testss)


@app.route('/camera',methods=['GET','POST'])
def open_cam():

    return render_template('camera.html')


def gen(camera):

    while True:
        frame = camera.get_frame(static_tracker.tracker)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera("a.mp4")),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam_feed')
def cam_feed():
    return Response(gen(VideoCamera(0)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='localhost', debug=True)