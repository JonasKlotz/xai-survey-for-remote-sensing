import os
from typing import Any

import torch
import torchmetrics
from matplotlib import pyplot as plt
from torchmetrics import (
    AveragePrecision,
    Accuracy,
    F1Score,
    MetricCollection,
)


class TrainingMetricsManager(torchmetrics.Metric):
    def __init__(self, config: dict, **kwargs: Any):
        super().__init__(**kwargs)
        self.task = config["task"]
        self.num_classes = config["num_classes"]
        self.config = config
        self._init_metrics()

        self.save_path = f"{config.get('training_root_path')}/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def __call__(self, stage="train", *args, **kwargs):
        return self._get_tracker_for_stage(*args, **kwargs)

    def __repr__(self):
        return self.val_metrics.__repr__()

    def _init_metrics(self):
        # self.confmat = torchmetrics.ConfusionMatrix(task=self.task, num_classes=self.num_classes, num_labels=self.num_classes)
        # self.roc = torchmetrics.ROC(self.task, num_classes=self.num_classes, num_labels=self.num_classes)
        metrics = MetricCollection(
            {
                "f1_score_macro": F1Score(
                    task=self.task,
                    average="macro",
                    threshold=0.5,
                    num_classes=self.num_classes,
                    num_labels=self.num_classes,
                ).to(self.config["device"]),
                "f1_score_micro": F1Score(
                    task=self.task,
                    average="micro",
                    threshold=0.5,
                    num_classes=self.num_classes,
                    num_labels=self.num_classes,
                ).to(self.config["device"]),
                "average_precision_macro": AveragePrecision(
                    task=self.task,
                    average="macro",
                    num_classes=self.num_classes,
                    num_labels=self.num_classes,
                ).to(self.config["device"]),
                "accuracy": Accuracy(
                    task=self.task,
                    threshold=0.5,
                    num_classes=self.num_classes,
                    num_labels=self.num_classes,
                ).to(self.config["device"]),
                # "confmat": self.confmat,
                # "roc": self.roc,
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.train_metrics = MetricCollection(
            {
                "accuracy": Accuracy(
                    task=self.task,
                    threshold=0.5,
                    num_classes=self.num_classes,
                    num_labels=self.num_classes,
                ).to(self.config["device"])
            }
        )

        self.val_tracker = torchmetrics.wrappers.MetricTracker(self.val_metrics)
        self.train_tracker = torchmetrics.wrappers.MetricTracker(self.train_metrics)
        self.test_tracker = torchmetrics.wrappers.MetricTracker(self.test_metrics)

    def compute(self, stage="train"):
        return self._get_tracker_for_stage(stage).compute()

    def update(self, preds: torch.Tensor, target: torch.Tensor = None, stage="train"):
        return self._get_tracker_for_stage(stage).update(preds, target)

    def reset(self, stage="train"):
        return self._get_tracker_for_stage(stage).reset_all()

    def plot(self, stage="test", *args, **kwargs):
        tracker = self._get_tracker_for_stage(stage)
        all_results = tracker.compute_all()

        fig, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=100)

        tracker.plot(val=all_results, ax=ax)
        if self.save_path is not None:
            filename = f"{self.config.get('experiment_name')}_{stage}.png"
            save_path = os.path.join(self.save_path, filename)
            plt.savefig(save_path)
        # plt.show()

    def _get_tracker_for_stage(self, stage="train"):
        if stage == "train":
            return self.train_tracker
        elif stage == "val":
            return self.val_tracker
        elif stage == "test":
            return self.test_tracker
        raise ValueError(f"Stage {stage} not supported.")

    def increment(self, stage="train"):
        self._get_tracker_for_stage(stage).increment()

    def compute_all(self, stage="train"):
        return self._get_tracker_for_stage(stage).compute_all()


if __name__ == "__main__":
    # test metricsManager
    config = {
        "task": "multiclass",
        "num_classes": 10,
        "save_path": "/home/jonasklotz/Studys/MASTERS/XAI/results/visualizations/train_plots/test.png",
    }
    metricsManager = TrainingMetricsManager(config)

    for _ in range(5):
        metricsManager.increment()
        for _ in range(5):
            y = torch.randint(10, (10,))
            y_hat = torch.randn(10, 10)
            metricsManager.update(y_hat, y)

    metricsManager.plot(stage="train")
