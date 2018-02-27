from flask import Flask, request, render_template
import pymysql

db = pymysql.connect("localhost", "root", "", "dynamic")

#api = Api(app)



def getdata():
    cursor = db.cursor()
    sql = "SELECT * FROM tracker_info"
    cursor.execute(sql)
    results = cursor.fetchall()
    return results
def insertdata(frame_count,trackers_list,restricted,entrance_info,time):
    f_number=str(frame_count)
    trackers_list=str(trackers_list)
    restricted=str(restricted)
    entrance_info=str(entrance_info)
    time=str(time)
    cursor = db.cursor()
    t = (f_number,trackers_list,restricted,entrance_info,time)
    cursor.execute("insert into tracker_info values(%s,%s,%s,%s,%s)", t)


