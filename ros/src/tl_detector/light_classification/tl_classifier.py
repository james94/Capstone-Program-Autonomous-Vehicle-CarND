from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import cv2
import rospy

TRAFFIC_LIGHTS = ['Green', 'Yellow', 'Red', 'Unknown']

class TLClassifier(object):
    def __init__(self, is_site):
        #TODO load classifier
        if is_site == True:
            path_to_pretrained_model = 'light_classification/models/site_model/'
        elif is_site == False:
            path_to_pretrained_model = 'light_classification/models/sim_model/'

        pretrained_model = path_to_pretrained_model + 'ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'

        # load pretrained model graph
        self.detection_graph = self.load_graph(pretrained_model)

        # extract tensors for detecting objects
        self.image_tensor, self.detection_boxes, \
        self.detection_scores, self.detection_classes = self.extract_tensors()
        self.sess = tf.Session(graph = self.detection_graph)

    def load_graph(self, graph_file):
        """ Loads frozen inference graph, the pretrained model """
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def extract_tensors(self):
        """ Extract relevant tensors for detecting objects """
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        return image_tensor, detection_boxes, detection_scores, detection_classes

    def filter_boxes(self, min_score, boxes, scores, classes):
        """ Return boxes with a confidence >= `min_score` """
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)
        
        filtered_boxes = boxes[idxs,  ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e. [0, 1]

        This converts it back to the original coordinate based on the image size 
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, scores):
        """ Draw bounding boxes on the image """
        box_bgr_color = (255, 0, 0)
        thickness = 2

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.8
        text_bgr_color = (200, 0, 0)
        line_type = cv2.LINE_AA
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            start_point = (left, top)
            end_point = (right, bot)
            cv2.rectangle(image, start_point, end_point, box_bgr_color, thickness)

            class_id = int(classes[i]) - 1
            light_prediction = str(int(scores[i]*100)) + '%'
            text = TRAFFIC_LIGHTS[class_id] + ': ' + light_prediction
            org = (left, int(top - 5))
            cv2.putText(image, text, org, font, fontScale, text_bgr_color, thickness, line_type)

    def get_classification(self, image, save_img):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        rospy.logwarn("get_classification called")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims( np.asarray(image_rgb, dtype=np.uint8), 0)

        with tf.Session(graph = self.detection_graph) as sess:
            # Actual detection
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})

            rospy.logwarn("boxes = %s", boxes)
            rospy.logwarn("scores = %s", scores)
            rospy.logwarn("classes = %s", classes)

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.7
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)

        if save_img == True:
            height, width, _ = image.shape
            box_coords = self.to_image_coords(boxes, height, width)
            self.draw_boxes(image, box_coords, classes, scores)
            cv2.imwrite('Capstone-Program-Autonomous-Vehicle-CarND/docs/images/detection/traffic-light.jpg', image)

        # check if traffic lights were not detected, then return light state unknown
        rospy.logwarn("length of scores = %s", len(scores))
        if len(scores) <= 0:
            traffic_light_class_id = 4
            traffic_light_state = TrafficLight.UNKNOWN
            return traffic_light_state

        # traffic light detected, return light state for green, yellow, red classification
        traffic_light_class_id = int(classes[np.argmax(scores)])
        rospy.logwarn("traffic light class id = %s", traffic_light_class_id)

        if traffic_light_class_id == 1:
            traffic_light_state = TrafficLight.GREEN
        elif traffic_light_class_id == 2:
            traffic_light_state = TrafficLight.YELLOW
        elif traffic_light_class_id == 3:
            traffic_light_state = TrafficLight.RED

        rospy.logwarn("traffic_light_state = %s", traffic_light_state)
        return traffic_light_state
