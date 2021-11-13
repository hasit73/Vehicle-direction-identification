import cv2
import numpy as np
        
class YoloDetection():
    def __init__(self, model_path: str, config: str, classes: str, width: int, height: int,
                 scale=0.00392, thr=0.5, nms=0.4, backend=0,
                 framework=3,
                 target=0, mean=[0, 0, 0]):
        
        super(YoloDetection,self).__init__()
        choices = ['caffe', 'tensorflow', 'torch', 'darknet']
        backends = (
            cv2.dnn.DNN_BACKEND_DEFAULT, cv2.dnn.DNN_BACKEND_HALIDE, cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE,
            cv2.dnn.DNN_BACKEND_OPENCV)
        targets = (
            cv2.dnn.DNN_TARGET_CPU, cv2.dnn.DNN_TARGET_OPENCL, cv2.dnn.DNN_TARGET_OPENCL_FP16, cv2.dnn.DNN_TARGET_MYRIAD)

        self.__confThreshold = thr
        self.__nmsThreshold = nms
        self.__mean = mean
        self.__scale = scale
        self.__width = width
        self.__height = height

        # Load a network
        self.__net = cv2.dnn.readNet(model_path, config, choices[framework])
        self.__net.setPreferableBackend(backends[backend])
        self.__net.setPreferableTarget(targets[target])
        self.__classes = None

        if classes:
            with open(classes, 'rt') as f:
                self.__classes = f.read().rstrip('\n').split('\n')


    def get_output_layers_name(self, net):
        all_layers_names = net.getLayerNames()
        return [all_layers_names[i-1] for i in net.getUnconnectedOutLayers()]

    def post_process_output(self, frame, outs):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        class_ids = []
        confidences = []
        boxes = []

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.__confThreshold:
                    center_x = int(detection[0] * frame_width)
                    center_y = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left = center_x - width / 2
                    top = center_y - height / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.__confThreshold, self.__nmsThreshold)
        return (indices, boxes, confidences, class_ids)

    def process_frame(self, frame: np.ndarray):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        blob = cv2.dnn.blobFromImage(frame, self.__scale, (self.__width, self.__height), self.__mean, True, crop=False)

        # Run a model
        self.__net.setInput(blob)
        outs = self.__net.forward(self.get_output_layers_name(self.__net))
        (indices, boxes, confidences, class_ids) = self.post_process_output(frame, outs)
        detected_objects = []

        for i in indices:
            
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            x = int(left)
            y = int(top)
            nw = int(width)
            nh = int(height)
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x + nw > frame_width:
                nw = frame_width - x
            if y + nh > frame_height:
                nh = frame_height - y
            detected_objects.append([self.__classes[class_ids[i]], x, y, nw, nh, confidences[i]])
        return detected_objects