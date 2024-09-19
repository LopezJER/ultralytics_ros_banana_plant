#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import rospy
import roslib.packages
import cv_bridge
import numpy as np
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D
from ultralytics_ros.msg import DetectionWithTrackID, YoloResult
from ultralytics import YOLO  # Make sure to import YOLO from your library
from vision_msgs.msg import ObjectHypothesisWithPose  # Import this if needed

class TrackerNode:
    def __init__(self):
        print("Initializing tracker node")
        yolo_model_path = rospy.get_param("~yolo_model_path", "yolov8n.pt")
        self.input_topic = rospy.get_param("~input_topic", "image_raw")
        self.result_topic = rospy.get_param("~result_topic", "yolo_result")
        self.result_image_topic = rospy.get_param("~result_image_topic", "yolo_image")
        self.conf_thres = rospy.get_param("~conf_thres", 0.25)
        self.iou_thres = rospy.get_param("~iou_thres", 0.45)
        self.max_det = rospy.get_param("~max_det", 300)
        self.classes = rospy.get_param("~classes", None)
        self.tracker = rospy.get_param("~tracker", "bytetrack.yaml")
        self.device = rospy.get_param("~device", None)
        self.result_conf = rospy.get_param("~result_conf", True)
        self.result_line_width = rospy.get_param("~result_line_width", None)
        self.result_font_size = rospy.get_param("~result_font_size", None)
        self.result_font = rospy.get_param("~result_font", "Arial.ttf")
        self.result_labels = rospy.get_param("~result_labels", True)
        self.result_boxes = rospy.get_param("~result_boxes", True)
        path = roslib.packages.get_pkg_dir("ultralytics_ros")
        self.model = YOLO(yolo_model_path)
        if yolo_model_path.split('.')[-1] != 'engine':
            self.model.fuse()
        self.sub = rospy.Subscriber(
            self.input_topic,
            Image,
            self.image_callback,
            queue_size=1,
            buff_size=2**24,
        )
        self.results_pub = rospy.Publisher(self.result_topic, YoloResult, queue_size=1)
        self.result_image_pub = rospy.Publisher(
            self.result_image_topic, Image, queue_size=1
        )
        self.bridge = cv_bridge.CvBridge()
        self.use_segmentation = yolo_model_path.endswith("-seg.pt")
        print(yolo_model_path)
        print(self.input_topic)
        
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        results = self.model.track(
            source=cv_image,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
            retina_masks=True,
            persist=True, 
        )

        if results is not None and results[0].boxes.id is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.header = msg.header
            yolo_result_image_msg.header = msg.header
            yolo_result_msg.detections_with_track_id = self.create_detections_with_track_id_array(results)
            yolo_result_image_msg = self.create_result_image(results)
            if self.use_segmentation:
                yolo_result_msg.masks = self.create_segmentation_masks(results)
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)
            
    def create_detections_with_track_id_array(self, results):
        detections_with_track_id_list = []
        bounding_boxes = results[0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_scores = results[0].boxes.conf
        track_ids = results[0].boxes.id

        rospy.loginfo(f"{bounding_boxes}, {classes}, {confidence_scores}, {track_ids}")

        for bbox, cls, conf, track_id in zip(bounding_boxes, classes, confidence_scores, track_ids):
            detection_with_track_id = DetectionWithTrackID()

            detection_with_track_id.detection.bbox.center.x = float(bbox[0])
            detection_with_track_id.detection.bbox.center.y = float(bbox[1])
            detection_with_track_id.detection.bbox.size_x = float(bbox[2])
            detection_with_track_id.detection.bbox.size_y = float(bbox[3])

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.id = int(cls)  # Class ID
            hypothesis.score = float(conf)  # Confidence score

            detection_with_track_id.detection.results.append(hypothesis)
            detection_with_track_id.track_id = int(track_id)  # Track ID

            detections_with_track_id_list.append(detection_with_track_id)

        return detections_with_track_id_list


    def create_result_image(self, results):
        plotted_image = results[0].plot(
            conf=self.result_conf,
            line_width=self.result_line_width,
            font_size=self.result_font_size,
            font=self.result_font,
            labels=self.result_labels,
            boxes=self.result_boxes,
        )
        result_image_msg = self.bridge.cv2_to_imgmsg(plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results):
        masks_msg = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    masks_msg.append(mask_image_msg)
        return masks_msg


if __name__ == "__main__":
    rospy.init_node("tracker_node")
    node = TrackerNode()
    rospy.spin()
