from sort import *

class SortAlgorithm:
    def __init__ (self):
        self.tracker = Sort()
        self.memory = {}
        self.trafficcounter = 0

    def set_line(self, line0, line1):
        self.line0 = line0
        self.line1 = line1      

    def intersect(self, A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def ccw(self, A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def output_counter(self, dets):
        ## Object Tracking
        tracks = self.tracker.update(dets)
        self.boxes = []
        self.indexIDs = []
        previous = self.memory.copy()
        self.memory = {}

        for track in tracks:
            self.boxes.append([track[0], track[1], track[2], track[3]])
            self.indexIDs.append(int(track[4]))
            self.memory[self.indexIDs[-1]] = self.boxes[-1]

        if len(self.boxes) > 0:
            i = int(0)
            for box in self.boxes:
                # extract the bounding box coordinates
                (x11, y11) = (int(box[0]), int(box[1]))
                (x12, y12) = (int(box[2]), int(box[3]))

                if self.indexIDs[i] in previous:
                    previous_box = previous[self.indexIDs[i]]
                    (x21, y21) = (int(previous_box[0]), int(previous_box[1]))
                    (x22, y22) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x11 + (x12-x11)/2), int(y11 + (y12-y11)/2))
                    p1 = (int(x21 + (x22-x21)/2), int(y21 + (y22-y21)/2))

                    if self.intersect(p0, p1, self.line0, self.line1): #line0 is line[0] & line1 is line[1]
                        self.trafficcounter += 1
                # display each bounding box/dot of the detected
                i += 1
        return self.trafficcounter