import numpy as np

def _true_rate(ground_truth, prediction, target_value):
    gt_target = ground_truth == target_value
    gt_num_target = np.count_nonzero(gt_target)

    if gt_num_target == 0:
        return 0.0

    pred_true_target = prediction[gt_target] == target_value
    pred_num_true_target = np.count_nonzero(pred_true_target)

    return float(pred_num_true_target) / gt_num_target


def true_positive_rate(ground_truth, prediction):
    return _true_rate(ground_truth, prediction, 1)


def true_negative_rate(ground_truth, prediction):
    return _true_rate(ground_truth, prediction, 0)


def false_positive_rate(ground_truth, prediction):
    return 1 - (_true_rate(ground_truth, prediction, 1))


def false_negative_rate(ground_truth, prediction):
    return 1 - (_true_rate(ground_truth, prediction, 0))
