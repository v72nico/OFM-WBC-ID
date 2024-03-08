from time import sleep
import matplotlib.pyplot as plt
import cv2
import torch
from math import inf, sqrt
import numpy as np

import openflexure_microscope_client as ofm_client
#from ofm_utils import capture_full_image

def pil_to_cv2(img):
    pil_image = img.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image.copy()
    return open_cv_image

class IdentificationHandler():
    def __init__(self):
        self.wbc_model = torch.hub.load('ultralytics/yolov5', "custom", path='bestv5.pt')
        self.diff_model = torch.hub.load('ultralytics/yolov5', "custom", path='bestdiffv5.pt')

    def identify_wbc(self, img):
        """Identifies wbcs using wbc_model"""
        results = self.wbc_model(img.copy())
        boxes = self.get_boxes(results)

        return boxes

    def differentiate_wbc(self, img):
        """Identifies and classifies wbcs using diff_model"""
        results = self.diff_model(img.copy())
        results.save()
        return results.pandas().xyxy[0]

    def get_boxes(self, results):
        """Converts model results to xyxy boxes"""
        boxes = results.xyxy[0]

        return boxes

class CenteringPlanner():
    """Plan movement to move between wbcs while maintaining a clear line of sight to the next wbc"""
    def __init__(self, x_margin, y_margin, tile_size_x, tile_size_y):
        self.identifier = IdentificationHandler()
        self.x_margin = x_margin
        self.y_margin = y_margin
        self.tile_size_x = tile_size_x
        self.tile_size_y = tile_size_y
        self.ordered_targets = []
        self.current_target = 0
        self.moves = 0
        self.confidence = 0.3

    def get_centers(self, boxes):
        """Converts xyxy box to coordinates of center point"""
        targets = []
        for box in boxes:
            y = int((box[1]+box[3])/2)
            x = int((box[0]+box[2])/2)
            targets.append((y,x))

        return targets

    def get_targets(self, boxes):
        """Converts boxes that meet the confidence threshold to center coordinates"""
        confident_boxes = []
        for box in boxes:
            if box[4] > self.confidence:
                confident_boxes.append(box)
        targets = self.get_centers(confident_boxes)

        return targets

    def identify_wbc(self, img):
        boxes = self.identifier.identify_wbc(img)
        target_wbcs = self.get_targets(boxes)
        clusters = self.cluster_targets(target_wbcs)
        cluster_targets = self.get_cluster_centers(clusters)

        return cluster_targets

    def get_closest_point(self, targets, target):
        dist = np.zeros(len(targets))
        for i in range(0, len(targets)):
            dist[i] = (sqrt((targets[i][0]-target[0])**2+(targets[i][1]-target[1])**2))
        return targets[np.argmin(dist)]

    def plan_path(self, img):
        targets = self.identify_wbc(img)
        self.ordered_targets = self.order_targets(targets)
        self.current_target = 0
        self.moves = 0

    def next_move(self, img):
        if len(self.ordered_targets) > self.current_target:
            targets = self.identify_wbc(img)
            center_targets = self.get_center_targets(targets)
            self.moves += 1
            if self.moves-1 > 0:
                if len(center_targets) > 0:
                    return self.return_centered()
                else:
                    return self.return_line_of_sight(targets)
            else:
                if self.current_target == 0:
                    if len(center_targets) > 0:
                        return self.return_centered()
                    return 'Uncentered', self.ordered_targets[self.current_target]
                else:
                    rel_pos = self.relative_position(self.ordered_targets[self.current_target-1], self.ordered_targets[self.current_target])
                    return 'Uncentered', rel_pos
        else:
            return 'Complete', []

    def return_centered(self):
        self.current_target += 1
        self.moves = 0
        return 'Centered', []

    def return_line_of_sight(self, targets):
        los_targets = self.get_line_of_sight_targets(targets)

        position = (self.tile_size_y/2, self.tile_size_x/2)
        if len(los_targets) > 0:
            close_target = self.get_closest_point(los_targets, position)
            return 'Uncentered', close_target
        else:
            #Unable to find next wbc in line of sight
            return 'Complete', []

    def relative_position(self, point_1, point_2):
        y_change = self.tile_size_y/2 - point_1[0]
        x_change = self.tile_size_x/2 - point_1[1]

        y = y_change + point_2[0]
        x = x_change + point_2[1]

        return [y, x]


    def get_slope(self, point_1, point_2):
        try:
            slope = (point_1[0] - point_2[0])/(point_1[1] - point_2[1])
        except ZeroDivisionError as e:
            slope = 0.001

        return slope

    def get_line_of_sight_targets(self, targets):
        ##TODO FUNCTION INCOMPREHNSIPLE
        target = self.ordered_targets[self.current_target]
        if self.current_target == 0:
            position = [self.tile_size_y/2, self.tile_size_x/2]
        else:
            position = self.ordered_targets[self.current_target-1]
        slope = self.get_slope(position, target)
        #TODO FIXX
        los_targets = []
        for this_target in targets:
            o_y = this_target[0]
            o_x = this_target[1]
            o_m = slope
            k = 20000
            y = o_y-self.tile_size_y/2
            x = o_x-self.tile_size_x/2
            y0 = target[0]-position[0]#-self.tile_size_y/2
            x0 = target[1]-position[1]#-self.tile_size_x/2
            m = slope#y0/x0
            a = (sqrt(k*y0**2*(x0**2+y0**2))+2*x0**3+2*x0*y0**2)/(2*(x0**2+y0**2))
            b = y0-((k*x0*y0)/(2*sqrt(k*y0**2*(x0**2+y0**2))))
            t = (x0-a)+x0
            v = (y0-b)+y0
            if m*(x-t)+v > y > m*(x-a)+b or m*(x-t)+v < y < m*(x-a)+b:
                if (target[0] > (-1/o_m)*(target[1]-self.tile_size_x/2)+self.tile_size_y/2):
                    if (o_y > (-1/o_m)*(o_x-self.tile_size_x/2)+self.tile_size_y/2):
                        los_targets.append(this_target)
                else:
                    if (o_y < (-1/o_m)*(o_x-self.tile_size_x/2)+self.tile_size_y/2):
                        los_targets.append(this_target)

        return los_targets

    def clear_sight(self, position, target, targets):
        m = self.get_slope(position, target)
        for other_target in targets:
            x = other_target[1]
            y = other_target[0]
            k = 50
            if m*(x-(target[1]+k))+(target[1]+k) > y > m*(x-(target[1]-k))+(target[1]-k):
                return False
        return True

    def get_center_targets(self, targets):
        y_center = self.tile_size_y/2
        x_center = self.tile_size_x/2

        size_factor = 4

        upper_y = int(y_center + self.y_margin/size_factor)
        lower_y = int(y_center - self.y_margin/size_factor)
        upper_x = int(x_center + self.x_margin/size_factor)
        lower_x = int(x_center - self.x_margin/size_factor)

        center_targets = []

        for target in targets:
            if upper_y > target[0] > lower_y:
                if upper_x > target[1] > lower_x:
                    center_targets.append(target)

        return center_targets

    def order_targets(self, targets):
        ordered_targets = []
        position = (self.tile_size_y/2, self.tile_size_x/2)
        all_targets = targets.copy()

        for i in range(0, len(targets)):
            close_target = self.get_closest_point(targets, position)
            if True:#TODO if self.clear_sight(position, close_target, all_targets):
                ordered_targets.append(close_target)
                position = close_target
                targets.remove(close_target)
            else:
                raise Exception("No line of sight")

        return ordered_targets

    def get_cluster_centers(self, clusters):
            points = []
            for cluster in clusters:
                min_x, max_x, min_y, max_y = self.make_rec(cluster)
                x = (min_x+max_x)/2
                y = (min_y+max_y)/2
                points.append((y,x,cluster))

            return points

    def make_rec(self, cluster):
        """Convert group of points to a bounding box"""
        max_x = 0
        max_y = 0
        min_x = inf
        min_y = inf
        for point in cluster:
            if point[1] > max_x:
                max_x = point[1]
            if point[1] < min_x:
                min_x = point[1]
            if point[0] > max_y:
                max_y = point[0]
            if point[0] < min_y:
                min_y = point[0]
        return min_x, max_x, min_y, max_y

    def cluster_targets(self, targets):
        #TODO MAKE LESS UGLY
        clusters = []
        for point in targets:
            if point in targets:
                this_cluster = [point]
                for other_point in targets:
                    if self.is_point_close(point, other_point):
                        targets.remove(other_point)
                        this_cluster.append(other_point)
                if len(this_cluster) > 1:
                    targets.remove(point)
                    clusters.append(this_cluster)

        for target in targets:
            clusters.append([target])
        return clusters

    def is_point_close(self, point, other_point):
        points_not_same = other_point != point
        within_y_margin = abs(point[0] - other_point[0]) < self.y_margin
        within_x_margin =  abs(point[1] - other_point[1]) < self.x_margin
        return points_not_same and within_y_margin and within_x_margin

class SlideCaptureHandler():
    def __init__(self, microscope_ip=None): #3280, 2464
        self.ip = microscope_ip

        self.tile_size_y = 624
        self.tile_size_x = 832
        self.tile_size_y_lrg = 2464
        self.tile_size_x_lrg = 3280
        self.resize_factor = self.tile_size_x_lrg/self.tile_size_x

        self.x_margin = 150
        self.y_margin = 150
        self.center_margin = 50

        self.tries = 5
        self.move_percent = 110

        self.centering_planner = CenteringPlanner(self.x_margin, self.y_margin, self.tile_size_x, self.tile_size_y)

        if self.ip == None:
            self.microscope = ofm_client.find_first_microscope()
        #else:
            #TODO connect to specific ip
        self.original_position = self.microscope.position.copy()

        self.spiral_direction = 0
        self.spiral_flip = False
        self.spiral_moves = 1
        self.spiral_completed_moves = 0

        self.move_queue = [0, 0]

        self.start_portion = 0.85
        self.portion_factor = 0.05

    def next_capture(self):

        imgs = self.capture_wbc_in_image()
        self.queue_spiral()
        self.go_queue()

        return imgs

    def move_to_original_position(self):
        """Move microsope back to self.original_position"""
        self.microscope.move(self.original_position)

    def go_queue(self):
        self.microscope.move_by_percent(self.move_queue[0], self.move_queue[1])
        self.move_queue = [0, 0]

    def next_direction(self):
        if self.spiral_direction == 3:
            self.spiral_direction = 0
        else:
            self.spiral_direction += 1

    def queue_spiral(self):
        move_percent = 110
        if self.spiral_direction == 0:
            self.move_queue[1] += move_percent
        elif self.spiral_direction == 1:
            self.move_queue[0] += move_percent
        elif self.spiral_direction == 2:
            self.move_queue[1] -= move_percent
        elif self.spiral_direction == 3:
            self.move_queue[0] -= move_percent

        self.spiral_completed_moves += 1

        print(self.spiral_completed_moves, '||||', self.spiral_direction, '||||', self.spiral_moves)

        if self.spiral_completed_moves == self.spiral_moves:
            self.switch_spiral_direction()

    def switch_spiral_direction(self):
        self.spiral_completed_moves = 0
        self.next_direction()

        if self.spiral_flip == True:
            self.spiral_moves += 1
            self.spiral_flip = False
        else:
            self.spiral_flip = True

    def get_target_percent(self, subdivsion):
        y_percent = ((subdivsion[0]-(self.tile_size_y/2))/self.tile_size_y)*100
        x_percent = ((subdivsion[1]-(self.tile_size_x/2))/self.tile_size_x)*100

        return y_percent, x_percent

    def queue_move(self, subdivsion, portion, reverse=False):
        y_percent, x_percent = self.get_target_percent(subdivsion)
        if reverse == True:
            y_percent = -y_percent
            x_percent = -x_percent

        self.move_queue[0] += y_percent*portion
        self.move_queue[1] += x_percent*portion

    def capture_wbc_in_image(self):
        """Centers and takes pictures of all wbcs in current field"""
        img = self.capture_images(takes=1)
        self.centering_planner.plan_path(img)
        tries = self.tries*len(self.centering_planner.ordered_targets)
        imgs = []
        portion = self.start_portion
        for i in range(0, tries):
            if i != 0:
                img = self.capture_images(takes=1, focus=False)
            status, target = self.centering_planner.next_move(img)
            if status == 'Centered':
                big_img = self.capture_images(size='full')
                centered_img = self.get_center_image(big_img)
                imgs.append(centered_img)
                portion = self.start_portion
            elif status == 'Complete':
                break
            elif status == 'Uncentered':
                self.queue_move(target, portion)
                portion += self.portion_factor
                self.go_queue()
        self.return_to_start()

        return imgs

    def return_to_start(self):
        ordered_targets = self.centering_planner.ordered_targets
        if len(ordered_targets) > 0:
            current_target = self.centering_planner.current_target-1
            target = ordered_targets[current_target]
            self.queue_move(target, 1, reverse=True)

    def get_center_image(self, img):
        y_sub_len = int((self.y_margin+self.center_margin)*self.resize_factor)
        x_sub_len = int((self.x_margin+self.center_margin)*self.resize_factor)
        y_center = self.tile_size_y_lrg/2
        x_center = self.tile_size_x_lrg/2

        upper_y = int(y_center + y_sub_len/2)
        lower_y = int(y_center - y_sub_len/2)
        upper_x = int(x_center + x_sub_len/2)
        lower_x = int(x_center - x_sub_len/2)

        return img[lower_y:upper_y, lower_x:upper_x]

    def 3ull_image(self):
        full_params = {
            "use_video_port": False,
            "temporary" : True,
            "filename": "py_capture"
        }
        self.microscope.capture_image_to_disk(params=full_params)
        id = self.microscope.list_capture_ids()[-1]
        self.microscope.download_from_id(id, '')
        img = cv2.imread('py_capture.jpeg')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def capture_images(self, takes=3, focus=True, size='small'):
        imgs = []
        if size == 'small':
            capture_function = self.microscope.grab_image_array
        else:
            capture_function = self.capture_full_image
        for i in range(0, takes):
            if focus:
                self.microscope.autofocus() #laplacian_autofocus({})
            img = capture_function()
            imgs.append(img)
        img = self.least_blurry(imgs)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def least_blurry(self, imgs):
        least_blurry = ''
        least_blurry_value = 0
        for img in imgs:
            #cv2img = pil_to_cv2(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplace = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplace > least_blurry_value:
                least_blurry = img
                least_blurry_value = laplace
        return least_blurry


class WbcCaptureHandler():
    def __init__(self, field_limit, wbc_limit):
        self.identifier = IdentificationHandler()
        self.cropped_img_margin = 15
        self.field_limit = field_limit
        self.wbc_limit = wbc_limit
        self.capture_handler = SlideCaptureHandler()

        self.fields = 0
        self.wbcs = 0
        self.mn = 0
        self.pmn = 0
        self.confidence = 0.5

    def capture_slide(self):
        """Move fields and capture wbc images until field limit is reached"""
        for i in range(0, self.field_limit):
            #if self.wbcs >= self.wbc_limit:
            #    break
            self.fields += 1
            imgs = self.capture_handler.next_capture()
            img_counter = 0
            for img in imgs:
                img_counter += 1
                # Temp
                cv2.imwrite(f'set/{self.fields}.{img_counter}.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                #
                self.classify_wbc(img)
                if self.wbcs != 0:
                    print('DIFFF:', 'PMN:', self.pmn/self.wbcs, 'MN:', self.mn/self.wbcs)
                print('PMN#:', self.pmn)
                print('MN#:',self.mn)
        self.capture_handler.move_to_original_position()

    def make_wbc_xyxy(self, result):
        """Convert results dict to xyxy list"""
        xyxy = []
        x_min = int(result['xmin'])
        xyxy.append(x_min)
        y_min = int(result['ymin'])
        xyxy.append(y_min)
        x_max = int(result['xmax'])
        xyxy.append(x_max)
        y_max = int(result['ymax'])
        xyxy.append(y_max)

        return xyxy

    def classify_wbc(self, img):
        results = self.identifier.differentiate_wbc(img)
        img_counter = 0
        for _, result in results.iterrows():
            img_counter += 1
            xyxy = self.make_wbc_xyxy(result)
            cropped_img = self.crop_img(xyxy, img)
            if result['confidence'] > self.confidence:
                self.wbcs += 1
                if result['class'] == 1:
                    self.pmn += 1
                    cv2.imwrite(f'show/pmn/{self.fields}.{img_counter}.jpg', cropped_img)
                if result['class'] == 2:
                    self.mn += 1
                    cv2.imwrite(f'show/mn/{self.fields}.{img_counter}.jpg', cropped_img)

    def crop_img(self, xyxy, img):
        height, length, _ = img.shape
        margin = self.cropped_img_margin
        x_min = max(xyxy[0]-margin, 0)
        x_max = min(xyxy[2]+margin, height)
        y_min = max(xyxy[1]-margin, 0)
        y_max = min(xyxy[3]+margin, length)
        cropped_img = img[y_min:y_max, x_min:x_max]
        cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

        return cropped_img

w = WbcCaptureHandler(200, 100)
w.capture_slide()
