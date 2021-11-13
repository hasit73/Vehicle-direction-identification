from vehicle_direction import find_angle_distance
from sort import *




class Tracker:
    CLASS_NAMES= ["bicycle","car","motorbike","aeroplane","bus","train","truck","boat","person"]

    def __init__(self):
        super(Tracker,self).__init__()
        
        self.car_tracker = Sort()
        self.detections = None
        self.__saved_track_ids = []
        self.frame = None
        self.trackers_centers = {}

    
    def track(self,output):
        dets = []
        track_results = []
        turn = "None"
        if(len(output)>0):
            for c in output:
                if(not c[0] in Tracker.CLASS_NAMES):
                    continue
                else:
                    x,y,w,h = c[1:5]
                    dets.append(np.array([x,y,(x+w),(y+h),c[5]]))
                    
            detts = np.array(dets)
            
            track_bbs_ids = self.car_tracker.update(detts)

            for d in track_bbs_ids :
                d = d.astype(np.int32)
                tid , x,y = str(d[4]) , d[0],d[1]
                w = d[2] - d[0]
                h = d[3] - d[1]
                if(not tid in self.trackers_centers):
                    self.trackers_centers[tid] = []
                self.trackers_centers[tid].append([x+w//2 , y+h//2])
                if(len(self.trackers_centers[tid]) >= 20):
                    turn = find_angle_distance(self.trackers_centers[tid])
                
                track_results.append({"track_id":tid,"points":[x,y,w,h],"class":"vehicle","track-dict":self.trackers_centers,"direction":turn})
                self.__saved_track_ids.append(tid)
                
        return track_results