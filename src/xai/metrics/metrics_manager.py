import datetime

import numpy as np
import torch

import src.xai.metrics.Quantus.quantus as quantus
from utility.csv_logger import CSVLogger
from xai.explanations.quantus_explanation_wrapper import explanation_wrapper
from xai.metrics.metrics_utiliies import (
    custom_aggregation_function,
    aggregate_continuity_metric,
    custom_aggregation_function_mlc,
)


class MetricsManager:
    def __init__(
        self,
        model: torch.nn.Module,
        explanation: callable,
        cfg: dict,
        metrics_config: dict = None,
        aggregate=True,
        log_dir=None,
        image_shape=(3, 224, 224),
        sentinel_value=np.nan,
        softmax=True,
    ):
        """
        Metrics Manager for evaluating metrics

        Parameters
        ----------
        model : torch.nn.Module
            model to evaluate
        aggregate : bool
            whether to aggregate the results of the metrics
        device : str
            device string for the model
        log : bool
            whether to log the results
        log_dir : str
            directory where the log file should be stored
        image_shape : tuple
            shape of the images
        sentinel_value : float
            value to return if an error occurs
        softmax : bool
            true if the model outputs softmax probabilities
        """
        self.model = model
        self.nr_runs = 10
        self.aggregate = aggregate

        self.metrics_config = metrics_config

        self.task = cfg["task"]
        self.device = cfg["device"]
        self.num_classes = cfg["num_classes"]

        self.disable_warnings = True

        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]

        self.patch_size = int(
            self.height / 8
        )  # todo: make this configurable? Also this can lead to errors?
        self.features_in_step = int(self.height / 8)
        self.num_samples = 5

        self.sentinel_value = sentinel_value
        self.softmax = softmax
        self.multi_label = self.task == "multilabel"

        self.categories = [
            "faithfulness",
            "robustness",
            "localization",
            "complexity",
            "randomization",
            "axiomatic",
        ]

        self.explain_func = explanation_wrapper
        self.explain_func_kwargs = {
            "explanation_method_name": explanation.attribution_name,
            "device": self.device,
            "multi_label": self.multi_label,
        }

        self.general_args = {
            "return_aggregate": self.aggregate,
            "disable_warnings": self.disable_warnings,
            "display_progressbar": False,
            "aggregate_func": custom_aggregation_function_mlc
            if self.multi_label
            else custom_aggregation_function,
            "multi_label": self.multi_label,
        }

        # load metrics
        self._load_metrics()

        column_names = []
        for category_name, metrics_category in self.metrics_config.items():
            for key in metrics_category.keys():
                column_names.append(key)

        self.log_dir = log_dir
        if self.log_dir:
            self.log = True
            self.metric_csv_logger = CSVLogger(
                log_dir=self.log_dir,
                filename=f"{explanation.attribution_name}_metrics",
                column_names=column_names,
            )
            self.time_csv_logger = CSVLogger(
                log_dir=self.log_dir,
                filename=f"{explanation.attribution_name}_time",
                column_names=column_names,
            )
            self.label_csv_logger = CSVLogger(
                log_dir=self.log_dir,
                filename=f"{explanation.attribution_name}_labels",
                column_names=["True Label", "Predicted Label"],
            )

    def _load_metrics(self):
        """Load all metrics"""
        self.categorized_metrics = {
            "Faithfulness": self._load_faithfulness_metrics(),
            "Robustness": self._load_robustness_metrics(),
            "Localization": self._load_localization_metrics(),
            "Complexity": self._load_complexity_metrics(),
            "Randomization": self._load_randomization_metrics(),
            "Axiomatic": self._load_axiomatic_metrics(),
        }
        # remove empty categories
        self.categorized_metrics = {
            k: v for k, v in self.categorized_metrics.items() if v is not None
        }

    def evaluate_batch(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray = None,
        y_true_batch: np.ndarray = None,
    ):
        all_results = {}
        time_spend = {}

        for category_name, metrics_category in self.categorized_metrics.items():
            results, time = self._evaluate_category(
                metrics_category, x_batch, y_batch, a_batch, s_batch
            )
            all_results.update(results)
            time_spend.update(time)

        if self.log:
            self.metric_csv_logger.update(all_results)
            self.time_csv_logger.update(time_spend)
            self.label_csv_logger.update(
                {"True Label": y_true_batch, "Predicted Label": y_batch}
            )

        return all_results, time_spend

    def _evaluate_category(
        self,
        metrics: dict,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray = None,
    ):
        """Evaluate a category of metrics

        Parameters
        ----------
        metrics : dict
            dictionary containing the metrics to evaluate
        x_batch : np.ndarray
            batch of images
        y_batch : np.ndarray
            batch of targets
        a_batch : np.ndarray
            batch of attributions
        s_batch: np.ndarray
            batch of segmentations

        Returns
        -------
        results : dict
            dictionary containing the results
        """
        results = {}
        time = {}
        already_unpacked_metrics = [
            "Relative Input Stability",
            "Relative Output Stability",
            "Relative Representation Stability",
            "Effective Complexity",
        ]
        for key in metrics.keys():
            start_time = datetime.datetime.now()
            res = metrics[key](
                model=self.model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=a_batch,
                s_batch=s_batch,
                device=self.device,
                softmax=True,
                channel_first=True,
                explain_func=self.explain_func,
                explain_func_kwargs=self.explain_func_kwargs,
            )
            if hasattr(metrics[key], "get_auc_score"):
                res = metrics[key].get_auc_score
            # the metrics that are not aggregated are already unpacked
            elif self.multi_label and key not in already_unpacked_metrics:
                res = res[0]  # batch unpacking
            time[key] = datetime.datetime.now() - start_time
            # print("Metric", key, "Result", res, "Time", time[key])
            results[key] = res
            if self.multi_label:
                assert_results_shape(res, y_batch)

        return results, time

    def _load_faithfulness_metrics(self):
        """Load all faithfulness metrics"""
        if "Faithfulness" not in self.metrics_config:
            return
        faithfulness_metrics = {
            "Faithfulness Correlation": quantus.FaithfulnessCorrelation(
                nr_runs=self.nr_runs, subset_size=self.height, **self.general_args
            ),
            "Faithfulness Estimate": quantus.FaithfulnessEstimate(
                features_in_step=self.features_in_step, **self.general_args
            ),
            "Monotonicity-Arya": quantus.Monotonicity(
                features_in_step=self.features_in_step, **self.general_args
            ),
            "Monotonicity-Nguyen": quantus.MonotonicityCorrelation(
                nr_samples=self.num_samples,
                features_in_step=self.features_in_step,
                **self.general_args,
            ),
            # We use AUC as aggregate function
            "Pixel-Flipping": quantus.PixelFlipping(
                features_in_step=self.features_in_step,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
            # We use AUC as aggregate function
            "Region Segmentation": quantus.RegionPerturbation(
                patch_size=self.patch_size,
                regions_evaluation=15,
                normalise=False,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
            # We use AUC as aggregate function
            "Selectivity": quantus.Selectivity(
                patch_size=self.patch_size,
                normalise=False,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
            "SensitivityN": quantus.SensitivityN(
                features_in_step=self.features_in_step,
                n_max_percentage=0.8,
                **self.general_args,
            ),
            "IROF": quantus.IROF(
                segmentation_method="slic", perturb_baseline="mean", **self.general_args
            ),
            "Infidelity": quantus.Infidelity(
                perturb_baseline="uniform",
                n_perturb_samples=5,
                perturb_patch_sizes=[self.patch_size],
                **self.general_args,
            ),
            "ROAD": quantus.ROAD(
                noise=0.01,
                perturb_func=quantus.noisy_linear_imputation,
                percentages=list(range(1, 50, 2)),
                **self.general_args,
            ),
            "Sufficiency": quantus.Sufficiency(threshold=0.6, **self.general_args),
        }
        return {
            k: v
            for k, v in faithfulness_metrics.items()
            if k in self.metrics_config["Faithfulness"]
        }

    def _load_robustness_metrics(self):
        """Load all robustness metrics"""
        if "Robustness" not in self.metrics_config:
            return
        robustness_metrics = {
            "Local Lipschitz Estimate": quantus.LocalLipschitzEstimate(
                nr_samples=self.num_samples,
                perturb_std=0.2,
                perturb_mean=0.0,
                **self.general_args,
            ),
            "Max-Sensitivity": quantus.MaxSensitivity(
                nr_samples=self.num_samples,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **self.general_args,
            ),
            "Avg-Sensitivity": quantus.AvgSensitivity(
                nr_samples=self.num_samples,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **self.general_args,
            ),
            "Continuity Test": quantus.Continuity(
                patch_size=self.patch_size,
                nr_steps=10,
                perturb_baseline="uniform",
                similarity_func=quantus.ssim,
                return_aggregate=self.aggregate,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                aggregate_func=aggregate_continuity_metric,
                # todo: Not sure if this is the right way for the aggregation
            ),
            "Consistency": quantus.Consistency(**self.general_args),
            "Relative Input Stability": quantus.RelativeInputStability(
                nr_samples=self.num_samples,
                return_aggregate=False,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
            "Relative Output Stability": quantus.RelativeOutputStability(
                nr_samples=self.num_samples,
                return_aggregate=False,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
            "Relative Representation Stability": quantus.RelativeRepresentationStability(
                nr_samples=self.num_samples,
                return_aggregate=False,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
        }
        return {
            k: v
            for k, v in robustness_metrics.items()
            if k in self.metrics_config["Robustness"]
        }

    def _load_localization_metrics(self):
        """Load all localization metrics"""
        if "Localisation" not in self.metrics_config:
            return
        localization_metrics = {
            "Focus": quantus.Focus(**self.general_args),
            "Pointing Game": quantus.PointingGame(**self.general_args),
            "Attribution Localisation": quantus.AttributionLocalisation(
                **self.general_args
            ),
            "Top-K Intersection": quantus.TopKIntersection(**self.general_args, k=100),
            "Relevance Rank Accuracy": quantus.RelevanceRankAccuracy(
                **self.general_args
            ),
            "Relevance Mass Accuracy": quantus.RelevanceMassAccuracy(
                **self.general_args
            ),
            "AUC": quantus.AUC(**self.general_args),
        }
        return {
            k: v
            for k, v in localization_metrics.items()
            if k in self.metrics_config["Localisation"]
        }

    def _load_randomization_metrics(self):
        """Load all randomization metrics"""
        if "Randomisation" not in self.metrics_config:
            return
        randomization_metrics = {
            "MPRT": quantus.MPRT(
                # layer_order="top_down",
                skip_layers=True,
                return_last_correlation=True,
                similarity_func=quantus.ssim,
                return_average_correlation=False,
                **self.general_args,
            ),
            "Random Logit": quantus.RandomLogit(
                num_classes=self.num_classes,
                similarity_func=quantus.ssim,
                **self.general_args,
            ),
        }
        return {
            k: v
            for k, v in randomization_metrics.items()
            if k in self.metrics_config["Randomisation"]
        }

    def _load_complexity_metrics(self):
        """Load all complexity metrics"""
        if "Complexity" not in self.metrics_config:
            return
        complexity_metrics = {
            "Sparseness": quantus.Sparseness(**self.general_args),
            "Complexity": quantus.Complexity(**self.general_args),
            "Effective Complexity": quantus.EffectiveComplexity(
                eps=0.3,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
            ),
        }
        return {
            k: v
            for k, v in complexity_metrics.items()
            if k in self.metrics_config["Complexity"]
        }

    def _load_axiomatic_metrics(self):
        """Load all axiomatic metrics"""
        if "Axiomatic" not in self.metrics_config:
            return
        axiomatic_metrics = {
            "Completeness": quantus.Completeness(
                return_aggregate=False,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                multi_label=self.multi_label,
                output_func=np.sum,
            ),
            "NonSensitivity": quantus.NonSensitivity(
                n_samples=1,
                features_in_step=self.height,  # here we need a high number as otherwise the metric is too slow
                perturb_baseline="black",
                perturb_func=quantus.baseline_replacement_by_indices,
                **self.general_args,
            ),  # complexity for metric = n_samples*(h*w/features_in_step) * model predict time
            "InputInvariance": quantus.InputInvariance(**self.general_args),
        }
        return {
            k: v
            for k, v in axiomatic_metrics.items()
            if k in self.metrics_config["Axiomatic"]
        }


def assert_results_shape(results, labels):
    # batchsize 1 only
    results = np.array(results)
    labels = np.array(labels)
    assert (
        results.shape == labels.shape
    ), f"Results shape {results.shape} != labels shape {labels.shape}"
