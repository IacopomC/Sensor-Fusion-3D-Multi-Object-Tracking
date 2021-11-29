# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
matplotlib.use('wxagg')  # change backend so that figure maximizing works on Mac as well
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):
    """
    Compute various performance measures to assess object detection
    1. Compute Intersection Over Union (iou) and distance between centers to find best match between detection and label
    2. Compute number of positive detections, true positives, false negatives, false positives

    Parameters:
    detections (list): detected bounding boxes in image coordinates [id, x, y, z, height, width, length, yaw]
    labels (RepeatedCompositeContainer): set of information for each object
                                         [box {x, y, z, w, l, h, y}, metadata {speed, acceleration}, type, id]
    labels_valid (numpy array): set of flags determining which label is valid [False, True, False,...]
    min_iou (float): Intersection Over Union threshold

    Returns:
    det_performance (list): set of parameters to evaluate detection
                            [ious, center_devs, [all_positives, true_positives, false_negatives, false_positives]]
    """

    # find best detection for each valid label
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:  # exclude all labels from statistics which are not considered valid

            ##################
            # Compute intersection over union (iou) and distance between centers
            ##################

            # extract the four corners of the current label bounding-box
            box = label.box
            label_bbox = tools.compute_box_corners(box.center_x, box.center_y, box.width, box.length, box.heading)

            # loop over all detected objects
            for det in detections:

                # extract the four corners of the current detection
                det_bbox = tools.compute_box_corners(det[1], det[2], det[5], det[6], det[7])  # x, y, w, l, y

                # compute the center distance between label and detection bounding-box in x, y, and z
                dist_x = box.center_x - det[1]
                dist_y = box.center_y - det[2]
                dist_z = box.center_z - det[3]

                # compute the intersection over union (IOU) between label and detected bounding-box
                # https://codereview.stackexchange.com/questions/204017/intersection-over-union-for-rotated-rectangles
                label_pol = Polygon(label_bbox)
                det_pol = Polygon(det_bbox)

                iou = label_pol.intersection(det_pol).area / label_pol.union(det_pol).area

                # if IOU > min_iou, store [iou,dist_x, dist_y, dist_z] in matches_lab_det
                if iou >= min_iou:
                    matches_lab_det.append([iou, dist_x, dist_y, dist_z])

        # find best match and compute metrics
        if matches_lab_det:
            # retrieve entry with max iou in case of multiple candidates
            best_match = max(matches_lab_det, key=itemgetter(1))
            ious.append(best_match[0])
            center_devs.append(best_match[1:])

    ##################
    # Compute positives and negatives for precision/recall
    ##################

    # compute the total number of detections really present in the scene
    all_positives = labels_valid.sum()

    # computer the number of true positives (correctly detected objects)
    true_positives = len(ious)

    # compute the number of false negatives (missed objects)
    false_negatives = all_positives - true_positives

    # compute the number of false positives (misclassified objects)
    false_positives = len(detections) - true_positives

    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]

    return det_performance


def compute_performance_stats(det_performance_all):
    """
    Evaluate object detection performance based on all frames

    Parameters:
    det_performance_all (list): set of detection performance parameters for every frame
                                [[ious, center_devs, [all_positives, true_positives, false_negatives, false_positives]],...]

    Returns:
    None

    """

    # extract elements
    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])

    # extract the total number of positives, true positives, false negatives and false positives
    _, true_positives, false_negatives, false_positives = np.sum(pos_negs, axis=0)

    # compute precision
    precision = true_positives / (true_positives + false_positives)

    # compute recall
    recall = true_positives / (true_positives + false_negatives)

    print('precision = ' + str(precision) + ", recall = " + str(recall))

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)


    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()
