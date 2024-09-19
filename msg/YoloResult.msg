# File: msg/YoloResult.msg

std_msgs/Header header
# Replace Detection2DArray with the new custom message type
DetectionWithTrackID[] detections_with_track_id
sensor_msgs/Image[] masks

