import numpy as np
import quantus
import torch

from utility.csv_logger import CSVLogger
from xai.explanations.explanation_manager import explanation_wrapper


class MetricsManager:
    def __init__(
        self,
        model: torch.nn.Module,
        explanation: callable,
        aggregate=True,
        device_string=None,
        log=False,
        log_dir=None,
        image_shape=(1, 28, 28),
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
        device_string : str
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

        self.device_string = device_string
        self.log = log
        self.log_dir = log_dir
        if self.log:
            self.csv_logger = CSVLogger(log_dir=self.log_dir)

        self.disable_warnings = True

        self.channels = image_shape[0]
        self.height = image_shape[1]
        self.width = image_shape[2]

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
        }

        self.general_args = {
            "return_aggregate": self.aggregate,
            "disable_warnings": self.disable_warnings,
            "display_progressbar": False,
        }
        # load metrics
        self._load_metrics()

    def _load_metrics(self):
        """Load all metrics"""
        self._load_faithfulness_metrics()
        self._load_robustness_metrics()
        self._load_localization_metrics()
        self._load_complexity_metrics()
        self._load_randomization_metrics()
        self._load_axiomatic_metrics()

    def evaluate_batch(
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
            batch of segmentations


        Returns
        -------
        all_results : dict
            dictionary containing all results
        """
        all_results = {}

        all_metrics_categories = [
            self.faithfulness_metrics,
            self.robustness_metrics,
            self.localization_metrics,
            self.complexity_metrics,
            self.randomization_metrics,
            self.axiomatic_metrics,
        ]

        for metrics_category in all_metrics_categories:
            results = self._evaluate_category(
                metrics_category, x_batch, y_batch, a_batch, s_batch
            )
            all_results.update(results)

        if self.log:
            self.csv_logger.update(all_results)

        return all_results

    def evaluate_batch_mlc(
        self,
        x_batch: torch.tensor,
        y_batch: torch.tensor,
        a_batch: torch.tensor,
        s_batch: torch.tensor = None,
    ):
        x_batch = (
            x_batch.unsqueeze(1).expand(-1, 6, -1, -1, -1).reshape(-1, 3, 120, 120)
        )
        # expand s_batch
        if s_batch is not None:
            s_batch = (
                s_batch.unsqueeze(1)
                .expand(
                    -1,
                    6,
                    -1,
                    -1,
                )
                .reshape(-1, 1, 120, 120)
            )
        # get indices where y_batch is not 0 (i.e. the labels that are present) batchsize x 1 where 1 from 0 to 5
        _ = torch.where(y_batch != 0)[1]

        # flatten a batch and y batch
        y_batch = y_batch.flatten(start_dim=0, end_dim=1)
        a_batch = a_batch.flatten(start_dim=0, end_dim=1)

        # filter out all indices where the label is 0
        tmp_indices = torch.where(y_batch != 0)[0]
        y_batch = y_batch[tmp_indices]
        a_batch = a_batch[tmp_indices]
        x_batch = x_batch[tmp_indices]
        if s_batch is not None:
            s_batch = s_batch[tmp_indices]

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
        for key in metrics.keys():
            try:
                results[key] = metrics[key](
                    model=self.model,
                    x_batch=x_batch,
                    y_batch=y_batch,
                    a_batch=a_batch,
                    s_batch=s_batch,
                    device=self.device_string,
                    softmax=True,
                    explain_func=self.explain_func,
                    explain_func_kwargs=self.explain_func_kwargs,
                )
                print(f"{key}: {results[key]}")
            except Exception as e:
                print(f"Error while evaluating {key}: {e}")
                results[key] = self.sentinel_value

        return results

    def _load_faithfulness_metrics(self):
        """Load all faithfulness metrics"""
        self.faithfulness_metrics = {
            "faithfulness_corr": quantus.FaithfulnessCorrelation(
                nr_runs=self.nr_runs, subset_size=224, **self.general_args
            ),
            "faithfulness_estimate": quantus.FaithfulnessEstimate(
                features_in_step=28, **self.general_args
            ),
            "monotonicity": quantus.Monotonicity(
                features_in_step=28, **self.general_args
            ),
            "monotonicity_correlation": quantus.MonotonicityCorrelation(
                nr_samples=10, features_in_step=28, **self.general_args
            ),
            "pixel_flipping": quantus.PixelFlipping(
                features_in_step=28, **self.general_args
            ),
            "region_perturb": quantus.RegionPerturbation(
                patch_size=4, regions_evaluation=10, normalise=True, **self.general_args
            ),
            "selectivity": quantus.Selectivity(patch_size=4, **self.general_args),
            "sensitivity_n": quantus.SensitivityN(
                features_in_step=28, n_max_percentage=0.8, **self.general_args
            ),
            "irof": quantus.IROF(
                segmentation_method="slic", perturb_baseline="mean", **self.general_args
            ),
            "infidelity": quantus.Infidelity(
                perturb_baseline="uniform",
                n_perturb_samples=5,
                perturb_patch_sizes=[4],
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

    def _load_robustness_metrics(self):
        """Load all robustness metrics"""
        self.robustness_metrics = {
            "local_lipschitz_estimate": quantus.LocalLipschitzEstimate(
                nr_samples=10, perturb_std=0.2, perturb_mean=0.0, **self.general_args
            ),
            "max_sensitivity": quantus.MaxSensitivity(
                nr_samples=10,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **self.general_args,
            ),
            "average_sensitivity": quantus.AvgSensitivity(
                nr_samples=10,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **self.general_args,
            ),
            "continuity": quantus.Continuity(
                patch_size=7,
                nr_steps=10,
                perturb_baseline="uniform",
                similarity_func=quantus.correlation_spearman,
                **self.general_args,
            ),
            "consistency": quantus.Consistency(**self.general_args),
            "relative_input_stability": quantus.RelativeInputStability(
                nr_samples=5, **self.general_args
            ),
            "relative_output_stability": quantus.RelativeOutputStability(
                nr_samples=5, **self.general_args
            ),
            "relative_representation_stability": quantus.RelativeRepresentationStability(
                nr_samples=5, **self.general_args
            ),
        }

    def _load_localization_metrics(self):
        """Load all localization metrics"""

        self.localization_metrics = {
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

    def _load_randomization_metrics(self):
        """Load all randomization metrics"""
        self.randomization_metrics = {
            "model_parameter_randomisation": quantus.MPRT(
                layer_order="bottom_up",
                similarity_func=quantus.correlation_spearman,
                return_average_correlation=True,
                **self.general_args,
            ),
            "random_logits": quantus.RandomLogit(
                num_classes=10, similarity_func=quantus.ssim, **self.general_args
            ),
        }

    def _load_complexity_metrics(self):
        """Load all complexity metrics"""
        self.complexity_metrics = {
            "sparseness": quantus.Sparseness(**self.general_args),
            "complexity": quantus.Complexity(**self.general_args),
            "effective_complexity": quantus.EffectiveComplexity(**self.general_args),
        }

    def _load_axiomatic_metrics(self):
        """Load all axiomatic metrics"""
        self.axiomatic_metrics = {
            "completeness": quantus.Completeness(**self.general_args),
            "non_sensitivity": quantus.NonSensitivity(
                n_samples=10,
                perturb_baseline="black",
                perturb_func=quantus.baseline_replacement_by_indices,
                **self.general_args,
            ),
            "input_invariance": quantus.InputInvariance(**self.general_args),
        }
