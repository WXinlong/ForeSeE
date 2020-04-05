import os
import numpy as np
from lib.utils.bounding_box import BoxList

class ObjectLabel:
    """Object Label Class
    1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                      'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                      'Misc' or 'DontCare'

    1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                      truncated refers to the object leaving image boundaries

    1    occluded     Integer (0,1,2,3) indicating occlusion state:
                      0 = fully visible, 1 = partly occluded
                      2 = largely occluded, 3 = unknown

    1    alpha        Observation angle of object, ranging [-pi..pi]

    4    bbox         2D bounding box of object in the image (0-based index):
                      contains left, top, right, bottom pixel coordinates

    3    dimensions   3D object dimensions: height, width, length (in meters)

    3    location     3D object location x,y,z in camera coordinates (in meters)

    1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]

    1    score        Only for results: Float, indicating confidence in
                      detection, needed for p/r curves, higher is better.
    """

    def __init__(self):
        self.type = ""  # Type of object
        self.truncation = 0.
        self.occlusion = 0.
        self.alpha = 0.
        self.x1 = 0.
        self.y1 = 0.
        self.x2 = 0.
        self.y2 = 0.
        self.h = 0.
        self.w = 0.
        self.l = 0.
        self.t = (0., 0., 0.)
        self.ry = 0.
        self.score = 0.

    def __eq__(self, other):
        """Compares the given object to the current ObjectLabel instance.

        :param other: object to compare to this instance against
        :return: True, if other and current instance is the same
        """
        if not isinstance(other, ObjectLabel):
            return False

        if self.__dict__ != other.__dict__:
            return False
        else:
            return True


def read_labels(label_path, results=False, ignore_cate=None):
    """Reads in label data file from Kitti Dataset.

    Returns:
    obj_list -- List of instances of class ObjectLabel.

    Keyword arguments:
    label_dir -- directory of the label files
    img_idx -- index of the image
    """

    # Define the object list
    obj_list = []

    # Extract the list
    if os.stat(label_path).st_size == 0:
        return

    if results:
        p = np.loadtxt(label_path, delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=16))
    else:
        p = np.loadtxt(label_path, delimiter=' ',
                       dtype=str,
                       usecols=np.arange(start=0, step=1, stop=15))

    # Check if the output is single dimensional or multi dimensional
    if len(p.shape) > 1:
        label_num = p.shape[0]
    else:
        label_num = 1

    for idx in np.arange(label_num):
        obj = ObjectLabel()

        if label_num > 1:
            # Fill in the object list
            obj.type = p[idx, 0]
            obj.truncation = float(p[idx, 1])
            obj.occlusion = float(p[idx, 2])
            obj.alpha = float(p[idx, 3])
            obj.x1 = float(p[idx, 4])
            obj.y1 = float(p[idx, 5])
            obj.x2 = float(p[idx, 6])
            obj.y2 = float(p[idx, 7])
            obj.h = float(p[idx, 8])
            obj.w = float(p[idx, 9])
            obj.l = float(p[idx, 10])
            obj.t = (float(p[idx, 11]), float(p[idx, 12]), float(p[idx, 13]))
            obj.ry = float(p[idx, 14])
            if results:
                obj.score = float(p[idx, 15])
            else:
                obj.score = 0.0
        else:
            # Fill in the object list
            obj.type = p[0]
            obj.truncation = float(p[1])
            obj.occlusion = float(p[2])
            obj.alpha = float(p[3])
            obj.x1 = float(p[4])
            obj.y1 = float(p[5])
            obj.x2 = float(p[6])
            obj.y2 = float(p[7])
            obj.h = float(p[8])
            obj.w = float(p[9])
            obj.l = float(p[10])
            obj.t = (float(p[11]), float(p[12]), float(p[13]))
            obj.ry = float(p[14])
            if results:
                obj.score = float(p[15])
            else:
                obj.score = 0.0

        if ignore_cate is not None and obj.type in ignore_cate:
            continue
        else:
            obj_list.append(obj)

    return obj_list


def rois2mask(rois, shape):
    mask = np.zeros(shape)
    for roi in rois:
        mask[int(roi.y1):int(roi.y2)+1, int(roi.x1):int(roi.x2)+1] = 1.

    return mask


def rois2mask_shrink(rois, shape):
    mask = np.zeros(shape)
    for roi in rois:
        h = int(roi.y2) - int(roi.y1) + 1
        w = int(roi.x2) - int(roi.x1) + 1
        
        mask[int(roi.y1)+int(h/4.):int(roi.y2)+1-int(h/4.), int(roi.x1)+int(w/4):int(roi.x2)+1-int(w/4)] = 1.

    return mask


def rois2mask_ins(rois, shape):
    np.random.shuffle(rois)
    mask = np.zeros(shape)
    for idx, roi in enumerate(rois):
        mask[int(roi.y1):int(roi.y2)+1, int(roi.x1):int(roi.x2)+1] = 1. + idx

    return mask
        

def rois2boxlist(rois, image_size, mode="xyxy"):
    boxes = []
    np.random.shuffle(rois)
    for roi in rois:
        xyxy = [int(roi.x1), int(roi.y1), int(roi.x2), int(roi.y2)]
        boxes.append(xyxy)

    boxlist = BoxList(boxes, image_size, mode=mode)
    return boxlist
    
