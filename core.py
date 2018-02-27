from collections import deque
import dlib
from datetime import datetime
from pytz import timezone
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from numba import jit
from Region_drawing import get_points
from Data_Access import testdb

from variables import static_variables

Frame_count = 0


##user defined variables may different with the situation
Sri_lanka = timezone('Asia/Colombo')
idx = 15#color ID
maximum_distance_to_restore_lost_tracker=120
tracke_deque_size=100



def tracker_moving_derection(tracker, tracker_locations):
    """
    this method is used to get the moving direction of a particular tracker.
     direction is tacker is taken w.r.t deque's last element and corrnt position
    :param tracker: coordinates of the tracker
    :param tracker_locations: coordinates list of the all trackers
    :return: left or right
    """
    tracker_deque = static_variables.tracker_queues_list[tracker_locations.index(tracker)]
    last_direction = direction([tracker_deque[-1][0], tracker_deque[-1][1]], [tracker_deque[0][0], tracker_deque[0][1]])
    return last_direction


def does_trackers_olap(test_tracker, tracker_coordinates):
    """
    this method is to ditermine wether trackers are overlapping or not
    :param test_tracker: coordinates of a given tracker
    :param tracker_coordinates: coordinates list of the all trackers
    :return: True or False
    """
    for tc in tracker_coordinates:
        if tc == test_tracker:
            continue
        if iou(tc, test_tracker) != 0:
            return [True, tc, test_tracker]
    return [False]


def count_people_in_reigon(rectangle, coordinates):
    """
    this method is to count people in a region
    :param rectangle: bounding box coordinates of region
    :param coordinates: coordinates list of detections
    :return: number of people
    """
    people_count_in_region = 0
    for c in coordinates:
        center = central(c)
        if (rectangle[0] < center[0] < rectangle[2]):
            people_count_in_region += 1

    return people_count_in_region


def is_inside_the_reigon(rectangle, tracker_coordinate):
    """
    this method is to define whether a tracker is in a given region
    :param rectangle: bounding box coordinates of region
    :param tracker_coordinate: location of a tracker
    :return: True or False
    """
    center = central(tracker_coordinate)
    if (rectangle[0] < center[0] < rectangle[2]):
        return True

    else:
        return False


def add_to_lost_trackers(detection_coordinates):
    """
    This method add a detection coordinates to lost trackers by checking it is already lost or not
    :param detection_coordinates: coordinates of detection
    :return: None
    """
    already_lost = False
    for lt in static_variables.lost_trackers:
        if (lt[0] == detection_coordinates[0]):
            already_lost = True
            lt[1] = detection_coordinates[1]
    if (already_lost == False):
        static_variables.lost_trackers.append(detection_coordinates)


def drawline(frame, x):
    """
    This method is to draw a line in the videofeed at given x value
    :param frame: frame of video feed
    :param x: x value
    :return: None
    """
    cv2.line(frame, (x, 0), (x, 500), (255, 100, 100), 1)


def distance(first, second):
    """
    This method determine the distance between two centrals
    :param first: (x,y) of first central point
    :param second: (x,y) of first central point
    :return: distance
    """
    dis = np.math.sqrt(((first[0] - second[0]) ** 2) + ((first[1] - second[1]) ** 2))
    return dis


def direction(first, second):
    """
    This method is determine second coordinate is in left or right to the first

    :param first: first coordinate
    :param second: second coordinate
    :return: left or right
    """
    if (first[0] < second[0]):
        return "right"
    else:
        return "left"


def central(rect):
    """
    This method is to get the central of given bounding box
    :param rect: coordinates of bounding box
    :return: [x,y] of central
    """
    x1 = rect[0]
    y1 = rect[1]
    x2 = rect[2]
    y2 = rect[3]
    cenx = min(x1, x2) + abs(x1 - x2) / 2
    ceny = min(y1, y2) + abs(y1 - y2) / 2
    return [cenx, ceny]


@jit
def iou(bb_test, bb_gt):
    """
    This method is computing the overlapping percentage of given 2 bounding boxes by intersection ove union
    :param bb_test: first bounding box coordinate
    :param bb_gt: second bounding box coordinate
    :return: percentage of overlapping
    """
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
              + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
    return (o * 100)


def is_walking(tracker_deque):
    """
    This method determine whether given tracker is walking or not.
    this checks the distance from current position and position of 50 (or last of deque) frames back and if that distance > 5
    determine it is walking tracker
    :param tracker_deque: queue of previous set of centrals of a tracker
    :return: True of False
    """
    if (len(tracker_deque) > 50):
        walk = distance([tracker_deque[50][0], tracker_deque[50][1]], [tracker_deque[50][0], tracker_deque[0][1]])
        if (walk < 5):
            return False
        else:
            return True
    else:
        walk = distance([tracker_deque[-1][0], tracker_deque[-1][1]], [tracker_deque[-1][0], tracker_deque[0][1]])
        if (walk < 5):
            return False
        else:
            return True

def remove_tracker(track_index, tracker_locations):
    """
    This method is to remove a tracker
    :param track_index: coordinates of the tracker want to remove
    :param tracker_locations: list of coordinates of all trackers
    :return: None
    """
    if track_index < len(static_variables.trackers_list):
        print "removing lost", static_variables.lost_trackers, track_index
        print "removed", track_index
        # trackers_list.remove(trackers_list[track_index])
        # tracker_queues_list.remove(tracker_queues_list[track_index])
        for ltrs in static_variables.lost_trackers:
            if ltrs[0] == track_index:
                static_variables.lost_trackers.remove(ltrs)
                break
        tracker_locations.remove(tracker_locations[track_index])


def add_Tracker(frame, startX, startY, endX, endY):
    """
    This method is to add new tracker
    :param frame: Frame of the video feed
    :param startX: bottom X of bounding box
    :param startY: bottom Y of bounding box
    :param endX: Upper X of bounding box
    :param endY: Upper Y of bounding box
    :return: None
    """
    tt = dlib.correlation_tracker()
    tt.start_track(frame, dlib.rectangle(startX, startY, endX, endY))

    static_variables.trackers_list.append(tt)
    static_variables.tracker_queues_list.append(deque(maxlen=tracke_deque_size))


def tracker_reformation(tracker_locations, detected_coordinates, frame, Frame_count):
    """
    This method is to reform the trackers. In the sence it
    remove unwanted trackers which doesont have detections,
    Add trackers for new detections countered
    Calibrate trackers if go below predefined overlapping percentage

    :param tracker_locations: coordinates list of all trackers
    :param detected_coordinates: coordinates list of all detections
    :param frame: frame of given video feed
    :param Frame_count: frame number
    :return: None
    """
    percentages = []
    if len(tracker_locations) > len(detected_coordinates):
        ##if lifetime of a lost tracker is greater than 10 that is removed
        for ltr in static_variables.lost_trackers:
            if ((Frame_count - ltr[2]) > 10):
                remove_tracker(ltr[0], tracker_locations)

        # trackers gana wadi human detections walata wada
        # "a person leaves" or tracker lost
        max_overlaps = len(detected_coordinates)
        number_of_overlaps = 0
        percentages_list = []
        #this calculates number of overlaps between detections and trackers
        for k in detected_coordinates:
            for l in tracker_locations:
                if (iou(k, l) != 0):
                    number_of_overlaps += 1
        for k in tracker_locations:
            not_matching = True
            if (does_trackers_olap(k, tracker_locations)[0]):
                cv2.putText(frame, " overlaps", (70, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            for l in detected_coordinates:
                #here is check whether a tracker is lost by being inside another tracker
                if iou(k, l) != 0:
                    if (does_trackers_olap(k, tracker_locations)[0]):
                        print "3 are overlapping", l, does_trackers_olap(k, tracker_locations)[1], does_trackers_olap(k, tracker_locations)[2]
                        mapdis1 = distance(central(does_trackers_olap(k, tracker_locations)[1]), central(l))
                        mapdis2 = distance(central(does_trackers_olap(k, tracker_locations)[2]), central(l))
                        dir1 = tracker_moving_derection(does_trackers_olap(k, tracker_locations)[1], tracker_locations)
                        dir2 = tracker_moving_derection(does_trackers_olap(k, tracker_locations)[2], tracker_locations)
                        print "mapping distance", mapdis1, mapdis2
                        # if it is so,identify the tracker with minimum distance to detection as correct one
                        # and put otherone to lost trackers

                        if (mapdis1 < mapdis2):
                            lab1 = str(tracker_locations.index(does_trackers_olap(k, tracker_locations)[2])) + " is lost"
                            cv2.putText(frame, lab1, (140, 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[10], 2)
                            add_to_lost_trackers(
                                [tracker_locations.index(does_trackers_olap(k, tracker_locations)[1]), does_trackers_olap(k, tracker_locations)[1],
                                 Frame_count])
                        else:
                            lab2 = str(tracker_locations.index(does_trackers_olap(k, tracker_locations)[1])) + " is lost"
                            cv2.putText(frame, lab2, (140, 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[10], 2)
                            add_to_lost_trackers(
                                [tracker_locations.index(does_trackers_olap(k, tracker_locations)[2]), does_trackers_olap(k, tracker_locations)[2],
                                 Frame_count])


                    not_matching = False
                    perc1 = iou(k, l)
                    percentages_list.append(perc1)
                    #this is to calibrate tracker when it is under the overlapping percentage of 90
                    if 10 < perc1 < 90 and not (does_trackers_olap(k, tracker_locations)[0]):
                        print "calibrating"
                        new_tracker = dlib.correlation_tracker()
                        new_tracker.start_track(frame, dlib.rectangle(l[0], l[1], l[2], l[3]))
                        try:
                            static_variables.trackers_list[tracker_locations.index(k)] = new_tracker
                        except IndexError:
                            print "error"

                    percentages.append(round(perc1, 2))


            if not_matching:
                # this tracker does not match with any of detections so it is added to lost trackers list

                add_to_lost_trackers([tracker_locations.index(k), k, Frame_count])
                # print "this tracker lost", k
        print number_of_overlaps, percentages_list
    else:

        max_overlaps = len(tracker_locations)
        number_of_overlaps = 0
        percentages_list = []
        #this calculate total number of overlaps between trackers and detections
        for k in detected_coordinates:
            for l in tracker_locations:
                if (iou(k, l) != 0):
                    number_of_overlaps += 1

        for k in detected_coordinates:
            not_matching = True
            is_lost = False;
            ptt = True
            for l in tracker_locations:
                if (does_trackers_olap(l, tracker_locations)[0]):
                    cv2.putText(frame, " overlaps", (70, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                if iou(k, l) != 0:
                    if (does_trackers_olap(l, tracker_locations)[0]):
                        print "3 are overlapping", k, does_trackers_olap(l, tracker_locations)[1], does_trackers_olap(l, tracker_locations)[2]
                        dir1 = str(tracker_locations.index(does_trackers_olap(l, tracker_locations)[1])) + " " + tracker_moving_derection(
                            does_trackers_olap(l, tracker_locations)[1], tracker_locations)
                        dir2 = str(tracker_locations.index(does_trackers_olap(l, tracker_locations)[2])) + " " + tracker_moving_derection(
                            does_trackers_olap(l, tracker_locations)[2], tracker_locations)
                        mapdis1 = distance(central(does_trackers_olap(l, tracker_locations)[1]), central(k))
                        mapdis2 = distance(central(does_trackers_olap(l, tracker_locations)[2]), central(k))

                        # here is check whether a tracker is lost by being inside another tracker
                        if (ptt):
                            ptt = False
                            # if it is so,identify the tracker with minimum distance to detection as correct one
                            # and put otherone to lost trackers
                            if (mapdis1 < mapdis2):
                                lab1 = str(tracker_locations.index(does_trackers_olap(l, tracker_locations)[2])) + " is lost"
                                cv2.putText(frame, lab1, (140, 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[10], 2)
                                add_to_lost_trackers(
                                    [tracker_locations.index(does_trackers_olap(l, tracker_locations)[1]), does_trackers_olap(l, tracker_locations)[1],
                                     Frame_count])
                            else:
                                lab2 = str(tracker_locations.index(does_trackers_olap(l, tracker_locations)[1])) + " is lost"
                                cv2.putText(frame, lab2, (140, 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[10], 2)
                                add_to_lost_trackers(
                                    [tracker_locations.index(does_trackers_olap(l, tracker_locations)[2]), does_trackers_olap(l, tracker_locations)[2],
                                     Frame_count])

                            cv2.putText(frame, dir1, (240, 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                            cv2.putText(frame, dir2, (240, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                    not_matching = False
                    perc1 = iou(k, l)
                    percentages_list.append(perc1)
                    if 10 < perc1 < 90 and not (does_trackers_olap(l, tracker_locations)[0]):
                        print "calibrating", number_of_overlaps
                        new_tracker = dlib.correlation_tracker()
                        new_tracker.start_track(frame, dlib.rectangle(k[0], k[1], k[2], k[3]))
                        static_variables.trackers_list[tracker_locations.index(l)] = new_tracker

                    percentages.append(round(perc1, 2))


            if not_matching:
                #this means this detection is not matching with any of tracker
                #so it is a detection without tracker
                #first try to map it with lost trackers
                print "detection is mismatching compare with lost trackers", static_variables.lost_trackers
                for ltr in static_variables.lost_trackers:
                    #if life of lost tracker is grater than 10 frames it is deleted
                    if ((Frame_count - ltr[2]) > 10):
                        remove_tracker(ltr[0], tracker_locations)
                    else:
                        tracker_deque = static_variables.tracker_queues_list[ltr[0]]
                        print(static_variables.tracker_queues_list[ltr[0]])
                        print(static_variables.tracker_queues_list[0])
                        #here is checks the directions of tracker previously moving and predict the position is sould be
                        detect_direction = direction([tracker_deque[0][0], tracker_deque[0][1]], central(k))
                        detect_distance = distance([tracker_deque[0][0], tracker_deque[0][1]], central(k))
                        last_direction = direction([tracker_deque[-1][0], tracker_deque[-1][1]], [tracker_deque[0][0], tracker_deque[0][1]])
                        direction_list = []
                        for u in range(1, len(tracker_deque) - 10):
                            current_direction = direction([tracker_deque[u + 10][0], tracker_deque[u + 10][1]], [tracker_deque[u][0], tracker_deque[u][1]])
                            info = [current_direction, [tracker_deque[u][0], tracker_deque[u][1]], [tracker_deque[0][0], tracker_deque[0][1]]]
                            direction_list.append(current_direction)

                        print direction_list
                        print "menna", ltr[0], detect_direction, last_direction, [tracker_deque[0][0], tracker_deque[0][1]], central(k), [
                            [tracker_deque[-1][0], tracker_deque[-1][1]], [tracker_deque[0][0], tracker_deque[0][1]]]

                        if (detect_direction == last_direction and detect_distance < maximum_distance_to_restore_lost_tracker):
                            #if directions are matching and distance is below predefined max
                            #that lost tracker is assigned to the lost detection and restored.
                            print detect_direction, last_direction
                            is_lost = True
                            new_tracker = dlib.correlation_tracker()
                            new_tracker.start_track(frame, dlib.rectangle(k[0], k[1], k[2], k[3]))
                            static_variables.trackers_list[ltr[0]] = new_tracker
                            print "tracker restored", ltr[0], ltr[1]
                            static_variables.lost_trackers.remove(ltr)#removing from lost trackers
                            break
                        static_variables.lost_detections.append(k)

                if (not (is_lost)):
                    #this is because that detection is not mapping with any of lost trackers so new tracker is
                    # needed to be initialized
                    print static_variables.lost_detections
                    if (len(detected_coordinates) > len(tracker_locations) + 1):
                        print "diga", detected_coordinates, tracker_locations
                        add_Tracker(frame, k[0], k[1], k[2], k[3])

                        print "new person came at", k
        print number_of_overlaps, percentages_list


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
#this caffe model can classify upto 20 objects
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
#create random list of colours
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("Caffe_Data/MobileNetSSD_deploy.prototxt.txt",
                               "Caffe_Data/MobileNetSSD_deploy.caffemodel")
# caffe= GoogLeNet trained network from Caffe model zoo.
# meke prototxt ekakui(architecture eka thiyenne meeke) model ekakui(meka train krla thyenne object detect krnna) denna oni

# meken return karanne net object ekak

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
fps = FPS().start()

class VideoCamera(object):

    def __init__(self, video):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(video)
        self.op = True
        self.Frame_count = 0
        static_variables.trackers_list = []
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

    def __del__(self):
        self.video.release()

    def get_frame(self):

        print "enter ", static_variables.entrance_information
        text_box = cv2.imread('pictures/text_background.png')
        self.Frame_count += 1
        print self.Frame_count
        # take frame by frame from video
        # change the width

        success, frame = self.video.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]
        if cv2.waitKey(1) == ord('p'):
            points = get_points.run(frame)
            if len(points) != 0:
                static_variables.restricted.append(points[0])

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),  # convert frame into a blob
                                     0.007843, (300, 300), 127.5)

        net.setInput(blob)  # me blob eka nural network ekata deela eke detections gannawa
        detections = net.forward()  # forward method eken return wenne <type 'numpy.ndarray'>.

        coordinates = []
        # print "detections"
        # loop over the detections
        for i in np.arange(0, detections.shape[
            2]):  # shape eken return wenne array eke dimention eka shape[2] kiyanne eke 3weni attribute
            #  eka detect kragatta objects gana
            # take the probability of each detection
            confidence = detections[0, 0, i, 2]
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence given
            if confidence > args["confidence"]:
                #
                idx = int(detections[0, 0, i, 1])  # get the index from detected object
                # 1 wenne i kiyana object eke wargaya class eke position eka
                if (idx == 15):  # filter only humans
                    box = detections[0, 0, i, 3:7] * np.array(
                        [w, h, w, h])  ##3:7 wenne i kiyana object eke x1=3,y1=4,x2=5,y2=6 points 4

                    (startX, startY, endX, endY) = box.astype("int")
                    # print [startX, startY, endX, endY],
                    coordinates.append([startX, startY, endX, endY])
                    # draw the prediction on the frame
                    # label = "{}: {:.2f}%".format(CLASSES[idx],
                    # confidence * 100)'
                    label = "person"

                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    yyy = startY - 15 if startY - 15 > 15 else startY + 15

                    # detection wala madde point eka
                    tup = [startX, startY, endX, endY]
                    xx = central(tup)[0]
                    yy = central(tup)[1]

                    cv2.line(frame, (xx, yy), (xx, yy), (255, 255, 255), 5)
                    # cv2.putText(frame, label, (startX, yyy),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        print "coordinates", coordinates

        people_count = str(len(coordinates))
        cv2.putText(text_box, "Number of people in the feed " + people_count, (50, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 2)
        cv2.putText(text_box, "Detected locations " + str(coordinates), (50, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 2)
        print
        print "trackers"
        tracker_locations = []



        for ii in xrange(len(static_variables.trackers_list)):
            static_variables.trackers_list[ii].update(frame)

            # Get the position of th object, draw a
            # bounding box around it and display it.
            rect = static_variables.trackers_list[ii].get_position()
            pt1 = (int(rect.left()), int(rect.top()))
            pt2 = (int(rect.right()), int(rect.bottom()))
            print pt1, pt2
            if (len(pt1) > 0 and len(pt2) > 0):
                static_variables.tracker_queues_list[ii].appendleft(
                    (central([pt1[0], pt1[1], pt2[0], pt2[1]])[0], central([pt1[0], pt1[1], pt2[0], pt2[1]])[1]))

            cv2.rectangle(frame, pt1, pt2, (255, 255, 255), 3)
            # print [int(rect.left()), int(rect.top()),int(rect.right()), int(rect.bottom())],
            tracker_locations.append([int(rect.left()), int(rect.top()), int(rect.right()), int(rect.bottom())])
            # pts.appendleft((central([pt1[0],pt1[1],pt2[0],pt2[1]])[0],central([pt1[0],pt1[1],pt2[0],pt2[1]])[1]))
            label = "tracker " + str(ii)
            # Get the position of the object, draw a
            # bounding box around it and display it.
            y = pt1[1] - 15 if pt1[1] - 15 > 15 else pt1[1] + 15
            cv2.putText(frame, label, (pt1[0], y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[15], 2)
        print tracker_locations
        cv2.putText(text_box, "Tracker locations " + str(tracker_locations), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 2)

        if (len(static_variables.restricted) != 0):
            for area in static_variables.restricted:
                p_count = count_people_in_reigon(area, coordinates)

                cv2.rectangle(frame, (area[0], area[1]), (area[2], area[3]), COLORS[2], 2)
                y = area[1] - 15 if area[1] - 15 > 15 else area[1] + 15
                region_index = static_variables.restricted.index(area)
                cv2.putText(frame, "Selected Region No : " + str(region_index), (area[0], y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[12], 2)

                cv2.putText(text_box, "Number of people in selected area " + str(region_index) + " : " + str(p_count),
                            (50, 200 + (region_index + 1) * 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 2)

                for tracker_coordinate in tracker_locations:
                    if is_inside_the_reigon(area, tracker_coordinate):
                        tracker_index = tracker_locations.index(tracker_coordinate)
                        is_new_entry = True
                        for entry in static_variables.entrance_information:
                            if entry[1] == tracker_index:
                                is_new_entry = False
                                break
                        if (is_new_entry):
                            # time_entered=strftime("%Y-%m-%d %H:%M:%S", gmtime())
                            time_entered = datetime.now(Sri_lanka)
                            entered_info = [region_index, tracker_index, time_entered]
                            static_variables.entrance_information.append(entered_info)
        for info in static_variables.entrance_information:
            cv2.putText(text_box,
                        "Tracker " + str(info[1]) + " entered to reigon " + str(info[0]) + " at " + str(info[2]),
                        (50, 215 + (static_variables.entrance_information.index(info) + 1) * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 1)

        # SI = max(0, min(XA2, XB2) - Max(XA1, XB1)) * Max(0, Min(YA2, YB2) - Max(YA1, YB1))
        print "list deka", static_variables.tracker_queues_list
        for queue in static_variables.tracker_queues_list:
            # loop over the set of tracked points
            index = static_variables.tracker_queues_list.index(queue)

            cv2.putText(frame, direction(queue[-1], queue[0]), (queue[0][0] - 10, queue[0][1] - 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            moving_direction = direction(queue[-1], queue[0]), (queue[0][0] - 10, queue[0][1] - 150)
            cv2.putText(text_box, "Tracker " + str(index) + " moving direction : " + str(moving_direction),
                        (50, ((index + 1) * 20) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 2)

            # print queue
            for i in xrange(1, len(queue)):
                # if either of the tracked points are None, ignore
                # them
                # if queue[i + 1] is None or queue[i] is None:
                #    continue

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                cv2.line(frame, queue[i - 1], queue[i], COLORS[static_variables.tracker_queues_list.index(queue)],
                         thickness)

                # print direction(queue[i-1], queue[i+5]),
            print
            # if(len(queue)>1):
            # print  "tracker ",tracker_queues_list.index(queue)," moves ",direction(queue[-1], queue[-2])
        tracker_reformation(tracker_locations, coordinates, frame, self.Frame_count)
        print
        cv2.putText(frame, str(self.Frame_count), (5, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[12], 1)
        cv2.putText(text_box, str(self.Frame_count), (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 1)
        # current_time=strftime("%Y-%m-%d %H:%M:%S", gmtime())
        current_time = datetime.now(Sri_lanka)
        static_variables.info_set=[]
        if (self.Frame_count % 20 == 0):
            for info in static_variables.entrance_information:
                str_time=str(info[2])[:10]
                static_variables.info_set.append("Tracker " + str(info[1]) + " entered to reigon " + str(info[0]) + " at " + str_time)
            testdb.insertdata(self.Frame_count, tracker_locations, static_variables.restricted,
                       static_variables.info_set, current_time)
        cv2.putText(text_box, str(current_time), (400, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 1)
        cv2.putText(text_box, "Selected Areas: " + str(static_variables.restricted), (50, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 0, 51), 1)

        # show the output frameq
        cv2.imshow("Frame", frame)
        # cv2.imshow("text",text_box)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        # break

        # update the FPS counter
        fps.update()
        if static_variables.camera_input==False:
            frame = np.hstack([frame, text_box])
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
