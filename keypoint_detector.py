#!/usr/bin/env python
#import rospy
#import rospkg

#from math import floor, ceil
#import cv2
#from os.path import join
import torch

#import message_filters
#from cv_bridge import CvBridge, CvBridgeError

#from sensor_msgs.msg import Image
#from rcta_object_pose_detection.msg import ira_dets

from models import StackedHourglass

class keypoint_detector_node(object):
    def __init__(self):
        #rospy.init_node('keypoint_detector_node')
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        #rospy.loginfo("Using %s", self.device)
        num_hourglasses = 2#rospy.get_param('~num_hourglasses', 2)
        hourglass_channels = 256#rospy.get_param('~hourglass_channels', 256)
        self.img_size = 256#rospy.get_param('~img_size', 256)

        #rospack = rospkg.RosPack()
        #model_base_path = join(rospack.get_path('rcta_object_pose_detection'), 
        #                  'models', 'keypoint_localization')
        model_path = '../logs/combined-half-995-400/checkpoints/2018_06_15-22_23_14.pt'
        #num_keypoints_file = rospy.get_param('~num_keypoints_file', 
        #    join(model_base_path, 'num_keypoints.txt'))
        #model_path = rospy.get_param('~model_path', join(model_base_path, "keypoints.pt"))

        #self.keypoints_indices = dict()
        #start_index = 0
        #with open(num_keypoints_file, 'r') as file:
        #    for line in file:
        #        split = line.split(' ')
        #        if len(split) == 2:
        #            self.keypoints_indices[split[0]] = (start_index, start_index + int(split[1]))
        #            start_index += int(split[1])
        #print "Keypoint indices:", self.keypoints_indices
        #rospy.loginfo("Ouput channels: %d", start_index)
        start_index = 30

        self.model = StackedHourglass(
            num_hg=num_hourglasses, 
            hg_channels=hourglass_channels, 
            out_channels=start_index).to(self.device)
        print (num_hourglasses, hourglass_channels)
        print (self.model.num_trainable_parameters())
        print (torch.load(model_path)['stacked_hg'])
        print (len(torch.load(model_path)['stacked_hg'].keys()))
        print (len(self.model.state_dict().keys()))

        self.model.load_state_dict(torch.load(model_path)['stacked_hg'])
        self.model.eval()
        print("Victory")
        exit()
        self.bridge = CvBridge()

        detections_sub = message_filters.Subscriber('detected_objects', ira_dets)
        image_sub = message_filters.Subscriber('camera/rgb/image_raw', Image)
        combined_sub = message_filters.ApproximateTimeSynchronizer([detections_sub, image_sub], 10, 1.0)
        combined_sub.registerCallback(self.detectKeypointsCb)
        rospy.loginfo("Spinning")
        rospy.spin()

    def detectKeypointsCb(self, detections_msg, image_msg):
        rospy.loginfo("Got %d detections in a %d x %d image", len(detections_msg.dets), image_msg.width, image_msg.height)

        try:
            image = self.bridge.imgmsg_to_cv2(image_msg, "passthrough")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        for detection in detections_msg.dets:
            x_min = floor(detection.bbox.points[0].x)
            y_min = floor(detection.bbox.points[0].y)

            x_max = ceil(detection.bbox.points[2].x)
            y_max = ceil(detection.bbox.points[2].y)
            patch = image[y_min:y_max, x_min:x_max, :]
            self.detectKeypointsFromPatch(patch, detection.obj_name)

    def detectKeypointsFromPatch(self, patch, object_class):
        resized_patch = cv2.resize(patch, (self.img_size, self.img_size))
        # cv2.imshow(object_class, resized_patch)
        # cv2.waitKey(1)

        tensor = torch.from_numpy(resized_patch).device(self.device)

        with torch.no_grad():
            pred_keypoints = self.model(tensor)

        print (pred_keypoints)

        # Select output


if __name__ == '__main__':
    keypoint_detector_node()
