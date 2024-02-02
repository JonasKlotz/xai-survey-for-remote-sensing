import datetime

import numpy as np
import quantus
import torch

from utility.csv_logger import CSVLogger
from xai.explanations.explanation_manager import explanation_wrapper
from xai.metrics.metrics_utiliies import (
    custom_aggregation_function,
    aggregate_continuity_metric,
)


class MetricsManager:
    def __init__(
        self,
        model: torch.nn.Module,
        explanation: callable,
        metrics_config: dict = None,
        aggregate=True,
        device=None,
        log=False,
        log_dir=None,
        image_shape=(3, 224, 224),
        sentinel_value=np.nan,
        softmax=True,
        num_classes=101,
        task="multiclass",
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
        self.task = task

        self.metrics_config = metrics_config

        self.device = device
        self.log = log
        self.log_dir = log_dir
        if self.log:
            self.csv_logger = CSVLogger(
                log_dir=self.log_dir, filename=explanation.attribution_name
            )

        self.disable_warnings = True

        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]

        self.patch_size = int(
            self.height / 8
        )  # todo: make this configurable? Also this can lead to errors?
        self.features_in_step = int(self.height / 8)
        self.num_samples = 1

        self.num_classes = num_classes

        self.sentinel_value = sentinel_value
        self.softmax = softmax

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
        }

        self.general_args = {
            "return_aggregate": self.aggregate,
            "disable_warnings": self.disable_warnings,
            "display_progressbar": False,
            "aggregate_func": custom_aggregation_function,
        }

        # load metrics
        self._load_metrics()

    def _load_metrics(self):
        """Load all metrics"""
        self.categorized_metrics = {
            "faithfulness": self._load_faithfulness_metrics(),
            "robustness": self._load_robustness_metrics(),
            "localization": self._load_localization_metrics(),
            "complexity": self._load_complexity_metrics(),
            "randomization": self._load_randomization_metrics(),
            "axiomatic": self._load_axiomatic_metrics(),
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
    ):
        if self.task == "multilabel":
            return self.evaluate_batch_mlc(x_batch, y_batch, a_batch, s_batch)
        else:
            return self.evaluate_batch_slc(x_batch, y_batch, a_batch, s_batch)

    def evaluate_batch_slc(
        self,
        x_batch: np.ndarray,
        y_batch: np.ndarray,
        a_batch: np.ndarray,
        s_batch: np.ndarray = None,
    ):
        """Evaluate a batch of images

        If self.log is true, the results are logged to a csv file

        Parameters
        ----------
        x_batch : np.ndarray
            batch of images
        y_batch : np.ndarray
            batch of targets
        a_batch : np.ndarray
            batch of attributions
        s_batch: np.ndarray
            batch of segmentations. Careful when the segmentation is None localization metrics will fail.


        Returns
        -------
        all_results : dict
            dictionary containing all results
        """
        all_results = {}
        time_spend = {}

        for category_name, metrics_category in self.categorized_metrics.items():
            results, time = self._evaluate_category(
                metrics_category, x_batch, y_batch, a_batch, s_batch
            )
            all_results.update(results)
            time_spend.update(time)

        if self.log:
            self.csv_logger.update(all_results)

        return all_results, time_spend

    def evaluate_batch_mlc(
        self,
        x_batch: torch.tensor,
        y_batch: torch.tensor,
        a_batch: torch.tensor,
        s_batch: torch.tensor = None,
    ):
        pass

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
        for key in metrics.keys():
            start_time = datetime.datetime.now()
            print()
            results[key] = metrics[key](
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
            time[key] = datetime.datetime.now() - start_time
            print(f"Time spent on {key}: {time[key]}\n" f"Results: {results[key]}")
            # except Exception as e:
            #     print(f"Error while evaluating {key}: {e}")
            #     results[key] = self.sentinel_value

        return results, time

    def _load_faithfulness_metrics(self):
        """Load all faithfulness metrics"""
        if "faithfulness" not in self.metrics_config:
            return
        faithfulness_metrics = {
            "faithfulness_corr": quantus.FaithfulnessCorrelation(
                nr_runs=self.nr_runs, subset_size=224, **self.general_args
            ),
            "faithfulness_estimate": quantus.FaithfulnessEstimate(
                features_in_step=self.features_in_step, **self.general_args
            ),
            "monotonicity": quantus.Monotonicity(
                features_in_step=self.features_in_step, **self.general_args
            ),
            "monotonicity_correlation": quantus.MonotonicityCorrelation(
                nr_samples=self.num_samples,
                features_in_step=self.features_in_step,
                **self.general_args,
            ),
            "pixel_flipping": quantus.PixelFlipping(
                features_in_step=self.features_in_step, **self.general_args
            ),
            "region_perturb": quantus.RegionPerturbation(
                patch_size=self.patch_size,
                regions_evaluation=10,
                normalise=True,
                **self.general_args,
            ),
            "selectivity": quantus.Selectivity(
                patch_size=self.patch_size, **self.general_args
            ),
            "sensitivity_n": quantus.SensitivityN(
                features_in_step=self.features_in_step,
                n_max_percentage=0.8,
                **self.general_args,
            ),
            "irof": quantus.IROF(
                segmentation_method="slic", perturb_baseline="mean", **self.general_args
            ),
            "infidelity": quantus.Infidelity(
                perturb_baseline="uniform",
                n_perturb_samples=5,
                perturb_patch_sizes=[self.patch_size],
                **self.general_args,
            ),
            "road": quantus.ROAD(
                noise=0.01,
                perturb_func=quantus.noisy_linear_imputation,
                percentages=list(range(1, 50, 2)),
                **self.general_args,
            ),
            "sufficiency": quantus.Sufficiency(threshold=0.6, **self.general_args),
        }
        return {
            k: v
            for k, v in faithfulness_metrics.items()
            if k in self.metrics_config["faithfulness"]
        }

    def _load_robustness_metrics(self):
        """Load all robustness metrics"""
        if "robustness" not in self.metrics_config:
            return
        robustness_metrics = {
            "local_lipschitz_estimate": quantus.LocalLipschitzEstimate(
                nr_samples=self.num_samples,
                perturb_std=0.2,
                perturb_mean=0.0,
                **self.general_args,
            ),
            "max_sensitivity": quantus.MaxSensitivity(
                nr_samples=self.num_samples,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **self.general_args,
            ),
            "average_sensitivity": quantus.AvgSensitivity(
                nr_samples=self.num_samples,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **self.general_args,
            ),
            "continuity": quantus.Continuity(
                patch_size=self.patch_size,
                nr_steps=10,
                perturb_baseline="uniform",
                similarity_func=quantus.ssim,
                return_aggregate=self.aggregate,
                disable_warnings=self.disable_warnings,
                display_progressbar=False,
                aggregate_func=aggregate_continuity_metric,  # todo: Not sure if this is the right way for the aggregation
            ),
            "consistency": quantus.Consistency(**self.general_args),
            "relative_input_stability": quantus.RelativeInputStability(
                nr_samples=self.num_samples, **self.general_args
            ),
            "relative_output_stability": quantus.RelativeOutputStability(
                nr_samples=self.num_samples, **self.general_args
            ),
            "relative_representation_stability": quantus.RelativeRepresentationStability(
                nr_samples=self.num_samples, **self.general_args
            ),
        }
        return {
            k: v
            for k, v in robustness_metrics.items()
            if k in self.metrics_config["robustness"]
        }

    def _load_localization_metrics(self):
        """Load all localization metrics"""
        if "localization" not in self.metrics_config:
            return
        localization_metrics = {
            "pointing_game": quantus.PointingGame(**self.general_args),
            "attribution_localisation": quantus.AttributionLocalisation(
                **self.general_args
            ),
            "top_k_intersection": quantus.TopKIntersection(**self.general_args, k=5),
            "relevance_rank_accuracy": quantus.RelevanceRankAccuracy(
                **self.general_args
            ),
            "relevance_mass_accuracy": quantus.RelevanceMassAccuracy(
                **self.general_args
            ),
            "auc": quantus.AUC(**self.general_args),
        }
        return {
            k: v
            for k, v in localization_metrics.items()
            if k in self.metrics_config["localization"]
        }

    def _load_randomization_metrics(self):
        """Load all randomization metrics"""
        if "randomization" not in self.metrics_config:
            return
        randomization_metrics = {
            "model_parameter_randomisation": quantus.MPRT(
                layer_order="bottom_up",
                similarity_func=quantus.ssim,
                return_average_correlation=True,
                **self.general_args,
            ),
            "random_logits": quantus.RandomLogit(
                num_classes=self.num_classes,
                similarity_func=quantus.ssim,
                **self.general_args,
            ),
        }
        return {
            k: v
            for k, v in randomization_metrics.items()
            if k in self.metrics_config["randomization"]
        }

    def _load_complexity_metrics(self):
        """Load all complexity metrics"""
        if "complexity" not in self.metrics_config:
            return
        complexity_metrics = {
            "sparseness": quantus.Sparseness(**self.general_args),
            "complexity": quantus.Complexity(**self.general_args),
            "effective_complexity": quantus.EffectiveComplexity(**self.general_args),
        }
        return {
            k: v
            for k, v in complexity_metrics.items()
            if k in self.metrics_config["complexity"]
        }

    def _load_axiomatic_metrics(self):
        """Load all axiomatic metrics"""
        if "axiomatic" not in self.metrics_config:
            return
        axiomatic_metrics = {
            "completeness": quantus.Completeness(**self.general_args),
            "non_sensitivity": quantus.NonSensitivity(
                n_samples=1,
                features_in_step=224,  # here we need a high number as otherwise the metric is too slow
                perturb_baseline="black",
                perturb_func=quantus.baseline_replacement_by_indices,
                **self.general_args,
            ),  # complexity for metric = n_samples*(h*w/features_in_step) * model predict time
            "input_invariance": quantus.InputInvariance(**self.general_args),
        }
        return {
            k: v
            for k, v in axiomatic_metrics.items()
            if k in self.metrics_config["axiomatic"]
        }
