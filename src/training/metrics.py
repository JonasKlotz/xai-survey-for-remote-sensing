import os
import sys
import logging
from datetime import datetime

import torch
from sklearn.metrics import f1_score, average_precision_score, accuracy_score


def create_logger(log_dir_path, logging_level=10):
    """
    Creates a logger with a time stamp output folder in
    :param logging_level: levels range from 50 - critical, 40 error .. to 10 - debug
    :param log_dir_path: log directory
    :return: logger
    """
    tick = datetime.now()

    log_dir_path += f'/log_{tick.strftime("%d.%m-%H:%M")}'

    logger = logging.getLogger("pytorch_logger")
    logger.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    os.makedirs(log_dir_path, exist_ok=True)
    output_file_handler = logging.FileHandler(os.path.join(log_dir_path, "run.log"))
    output_file_handler.setFormatter(formatter)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger


class MetricTracker(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 3)


def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate  both micro and macro for F1 and mAP score
    """
    y_pred_threshed = (y_pred > threshold).astype(torch.long)
    acc = round(accuracy_score(y_true=y_true, y_pred=y_pred), 3)
    mif1 = round(f1_score(y_true=y_true, y_pred=y_pred_threshed, average='micro'), 3)
    maf1 = round(f1_score(y_true=y_true, y_pred=y_pred_threshed, average='macro'), 3)
    miMAP = round(average_precision_score(y_true=y_true, y_score=y_pred, average='micro'), 3)
    maMAP = round(average_precision_score(y_true=y_true, y_score=y_pred, average='macro'), 3)
    return mif1, maf1, miMAP, maMAP, acc


class CSV_logger:
    """
    Class for logging metrics to a csv file
    """
    metric_names = ["epoch", "train_miAP", "train_maAP", "train_miF1", "train_maF1", "train_loss",
                    "valid_miAP", "valid_maAP", "valid_miF1", "valid_maF1", "valid_loss",
                    "test_miAP", "test_maAP", "test_miF1", "test_maF1", "test_loss"]

    def __init__(self, file_name, dir_name, metric_names=None):
        try:
            os.makedirs(dir_name)
        except OSError as exc:
            pass
        if metric_names:
            self.metric_names = metric_names

        self.file_path = os.path.join(dir_name, f'{file_name}.csv')
        csv_header = s = ",".join(self.metric_names)
        with open(self.file_path, 'a') as csv_file:
            csv_file.write(csv_header)
            csv_file.write('\n')

    @staticmethod
    def get_metric_names(metric_names):
        return metric_names

    def write_csv(self, metrics, new_line=False):
        with open(self.file_path, 'a') as csv_file:
            for m in metrics:
                csv_file.write(str(m) + ',')
            if new_line:
                csv_file.write('\n')
