
import dlib

def setTracker(trck):
    tracker=trck
def remove_tracker(track_index,tracker):
    if track_index < len(tracker):
        tracker.remove(tracker[track_index])

def add_Tracker(frame,startX, startY, endX, endY,tracker):
    tt=dlib.correlation_tracker()
    tt.start_track(frame, dlib.rectangle(startX, startY, endX, endY))

    tracker.append(tt)
def is_olap(k,l):
    hoverlaps = True
    voverlaps = True

    if (k[0] > l[2]) or (k[2] < l[0]):
        hoverlaps = False
    if (k[3] < l[1]) or (k[1] > l[3]):
        voverlaps = False
    if (hoverlaps and voverlaps):
        return True
def overlaps(trcks,coordinates,frame,tracker):
    percentages=[]
    if len(trcks)>len(coordinates):
        #"a person leaves"
        for k in trcks:
            not_matching = True
            for l in coordinates:
                if is_olap(k, l):
                    not_matching = False
                    area1 = abs(k[0] - k[2]) * abs(k[1] - k[3])

                    cod1 = abs(l[0] - l[2]) * abs(l[1] - l[3])

                    is1 = abs(max(k[0], l[0]) - min(k[2], l[2])) * abs(
                        max(k[1], l[1]) - min(k[3], l[3]))
                    perc1 = (float(is1) / (area1 + cod1 - is1)) * 100
                    # if perc1>75:
                    #   print "success",k,l
                    percentages.append(round(perc1, 2))
                    print "overlaping percentage", perc1, k, l
            if not_matching:
                remove_tracker(trcks.index(k),tracker)
                print "this tracker should remove", k
    else:
        #"new person enterd"
        for k in coordinates:
            not_matching = True
            for l in trcks:
                if is_olap(k, l):
                    not_matching = False
                    area1 = abs(k[0] - k[2]) * abs(k[1] - k[3])

                    cod1 = abs(l[0] - l[2]) * abs(l[1] - l[3])

                    is1 = abs(max(k[0], l[0]) - min(k[2], l[2])) * abs(
                        max(k[1], l[1]) - min(k[3], l[3]))
                    perc1 = (float(is1) / (area1 + cod1 - is1)) * 100
                    # if perc1>75:
                    #   print "success",k,l
                    percentages.append(round(perc1, 2))
                    print "overlaping percentage", perc1, k, l
            if not_matching:
                add_Tracker(frame,k[0],k[1],k[2],k[3],tracker)
                print "new person came at", k


