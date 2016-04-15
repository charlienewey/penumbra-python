import numpy as np

def _match(ground_truth, prediction, target_value):
    gt_target = ground_truth == target_value
    gt_num_target = np.count_nonzero(gt_target)

    pred_true_target = prediction[gt_target] == target_value
    pred_num_true_target = np.count_nonzero(pred_true_target)

    return pred_num_true_target

def _no_match(ground_truth, prediction, target_value):
    gt_target = ground_truth == target_value
    gt_num_target = np.count_nonzero(gt_target)

    pred_false_target = prediction[gt_target] != target_value
    pred_num_false_target = np.count_nonzero(pred_false_target)

    return pred_num_false_target


def true_positives(ground_truth, prediction):
    return _match(ground_truth, prediction, 1)


def true_negatives(ground_truth, prediction):
    return _match(ground_truth, prediction, 0)


def false_positives(ground_truth, prediction):
    return _no_match(ground_truth, prediction, 1)


def false_negatives(ground_truth, prediction):
    return _no_match(ground_truth, prediction, 0)
