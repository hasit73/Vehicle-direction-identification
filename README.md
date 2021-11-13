# Vehicle-direction-identification
Vehicle direction identification consists of three module detection , tracking and direction recognization.


### Algorithm used : Yolo algorithm for detection + SORT algorithm to track vehicles + vector based direction detection
### Backend : opencv and python
### Library required:

- opencv = '4.5.4-dev'
- scipy = '1.4.1'
- filterpy 
- lap
- scikit-image


### IMPORTANT:

- I hadn't uploaded model weights and configuration files (which were used for object detection) here because those were already available in yolo_detection repo
- download yolo tiny weights , config file and coco.names file from here : [https://github.com/hasit73/yolo_detection]
- For detection i was using same code which was available in yolo_detection repo.

# Quick Overview about structure

#### 1) main.py

- Loading model and user configurations
- perform io interfacing tasks


#### 2) yolo.py

- use opencv modules to detect objects from user given media(photo/video)
- detection take place inside this file


#### 3) config.json

- user configuration are mentioned inside this file
- for examples : input shapes and model parameters(weights file path , config file path etc) are added in config.json


#### 4) tracker.py

- it have one Tracker class that will be used to track vehicles.

#### 5) sort.py

- SORT algorithm implementations
- Kalman filter operations


#### 6) vehicle_direction.py

- Vector based direction recognization


# How to use 

1) clone this directory

 
2) use following command to run detection and tracking on your custom video

  ```
  python main.py -c config.json -v <media_path>
  ```

  Example: 
  ```
  python main.py -c config.json -v car1.mp4
  ```
  
- Note : Before executing this command make sure that you have downloaded model weights and config file for yolo object detection.

### Results


- output

https://user-images.githubusercontent.com/69752829/141656351-d2ce89be-5b6c-48fc-87ba-e3bd71283403.mp4



### Limitations:

There are few primary drawbacks of this appoach

1) direction recogization totally depends on detection and tracking.
2) if camera properly arranged then it gives accurate results (Suppose any object is in front of camera and come forward towards camera then it gives bad results)
    but if you try to use this approach in cctv suviellence then it gives satisfactory results.
    
3) in few cases , it performs bad, because right now it works on only single keypoint (center of object) we can improve its performace by detecting multiple keypoints and use majority votes result.
 
## If it's helful for you then please give star :)

