from yolo import YoloDetection
import cv2
import argparse
from tracker import Tracker
import time
import imageio

CONFIG_FILE = None
model = None



def load_config(config_path):
    global CONFIG_FILE
    CONFIG_FILE = eval(open(config_path).read())


def load_model():
    global model
    model = YoloDetection(CONFIG_FILE["model-parameters"]["model-weights"],
                    CONFIG_FILE["model-parameters"]["model-config"],
                    CONFIG_FILE["model-parameters"]["model-names"],
                    CONFIG_FILE["shape"][0],
                    CONFIG_FILE["shape"][1])


def start_detection(media_path):
    vehicle_tracker = Tracker()    
    cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(media_path)
    writer = imageio.get_writer("demo.mp4")
    ret = True
    while ret:
        ret , frame = cap.read()
        if(ret):
            st = time.time()
            detections = model.process_frame(frame)
            tracked_results = vehicle_tracker.track(detections)
            for r in tracked_results:
                x,y,w,h = r["points"]
                track_id = r["track_id"]
                cv2.rectangle(frame,(x,y),(x+w,y+h),thickness=2,color=(255,255,0))
                cv2.putText(frame,track_id+"-"+str(r["direction"]),(x,y-3),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
            et = time.time()
            cv2.putText(frame,f"FPS : {round(1/(et-st) , 2)}",(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
            
            cv2.imshow("Video",frame)
            writer.append_data(frame[:,:,::-1])
            key = cv2.waitKey(30)
            if(key==27):
                break
            if(key==32):
                cv2.waitKey(-1)
    
    writer.close()

    cv2.destroyAllWindows()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Give config file and media file path")
    parser.add_argument("--config","-c")
    parser.add_argument("--debug","-d")
    parser.add_argument("--video","-v")
    args = parser.parse_args()
    config_path = args.config
    load_config(config_path)
    load_model()
    start_detection(args.video)


