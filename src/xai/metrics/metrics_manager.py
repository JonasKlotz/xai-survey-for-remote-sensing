import numpy as np
import quantus


class MetricsManager:
    def __init__(self, model, aggregate=True, device_string=None, log=False, log_dir=None, image_shape=(1, 28, 28)):
        self.model = model
        self.nr_runs = 10
        self.aggregate = aggregate

        self.device_string = device_string
        self.log = log
        self.log_dir = log_dir
        if self.log:
            self.csv_logger = None  # .CSVLogger(self.log_dir)

        self.disable_warnings = True

        self.categories = ['faithfulness', 'robustness', 'localization', 'complexity', 'randomization', 'axiomatic']
        # load metrics
        self._load_faithfulness_metrics()
        self._load_robustness_metrics()
        self._load_localization_metrics()
        self._load_complexity_metrics()
        self._load_randomization_metrics()
        self._load_axiomatic_metrics()

    def evaluate_batch(self,
                       x_batch: np.ndarray,
                       y_batch: np.ndarray,
                       a_batch: np.ndarray,
                       s_batch: np.ndarray = None):
        """ Evaluate a batch of images

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

        """
        all_results = {}
        faithfulness_result = self._evaluate_category(self.faithfulness_metrics,
                                                      x_batch=x_batch,
                                                      y_batch=y_batch,
                                                      a_batch=a_batch,
                                                      s_batch=s_batch
                                                      )

        all_results.update(faithfulness_result)

        robustness_result = self._evaluate_category(self.robustness_metrics,
                                                    x_batch=x_batch,
                                                    y_batch=y_batch,
                                                    a_batch=a_batch,
                                                    s_batch=s_batch
                                                    )

        all_results.update(robustness_result)

        localization_result = self._evaluate_category(self.localization_metrics,
                                                      x_batch=x_batch,
                                                      y_batch=y_batch,
                                                      a_batch=a_batch,
                                                      s_batch=s_batch
                                                      )

        all_results.update(localization_result)

        if self.log:
            self.csv_logger.update(all_results)

        return all_results

    def _evaluate_category(self,
                           metrics: dict,
                           x_batch: np.ndarray,
                           y_batch: np.ndarray,
                           a_batch: np.ndarray,
                           s_batch: np.ndarray = None):
        result = {}
        for key in metrics.keys():
            try:
                result[key] = metrics[key](model=self.model,
                                           x_batch=x_batch,
                                           y_batch=y_batch,
                                           a_batch=a_batch,
                                           s_batch=s_batch,
                                           device=self.device_string,
                                           softmax=True
                                           )
            except Exception as e:
                print(f"Error while evaluating {key}: {e}")
                result[key] = None
            print(f"{key}: {result[key]}")

        return result

    def _load_faithfulness_metrics(self):
        ff_args = {'return_aggregate': self.aggregate,
                   'disable_warnings': self.disable_warnings,
                   'display_progressbar': False,
                   }

        self.faithfulness_metrics = {

            'faithfulness_corr': quantus.FaithfulnessCorrelation(
                nr_runs=self.nr_runs,
                subset_size=224,
                perturb_func=None,
                **ff_args

            ),
            'faithfulness_estimate': quantus.FaithfulnessEstimate(
                features_in_step=28,
                **ff_args
            ),

            'monotonicity': quantus.Monotonicity(
                features_in_step=28,
                **ff_args
            ),

            'monotonicity_correlation': quantus.MonotonicityCorrelation(
                nr_samples=10,
                features_in_step=28,
                **ff_args
            ),

            'pixel_flipping': quantus.PixelFlipping(
                features_in_step=28,
                **ff_args
            ),

            'region_perturb': quantus.RegionPerturbation(
                patch_size=4,
                regions_evaluation=10,
                normalise=True,
                **ff_args
            ),
            'selectivity': quantus.Selectivity(
                patch_size=4,
                **ff_args
            ),
            'sensitivity_n': quantus.SensitivityN(
                features_in_step=28,
                n_max_percentage=0.8,
                **ff_args
            ),
            'irof': quantus.IROF(
                segmentation_method="slic",
                perturb_baseline="mean",
                **ff_args
            ),
            'infidelity': quantus.Infidelity(
                perturb_baseline="uniform",
                n_perturb_samples=5,
                perturb_patch_sizes=[4],
                **ff_args
            ),
            'road': quantus.ROAD(
                noise=0.01,
                perturb_func=quantus.noisy_linear_imputation,
                percentages=list(range(1, 50, 2)),
                **ff_args
            ),
            'sufficiency': quantus.Sufficiency(
                threshold=0.6,
                **ff_args
            )
        }

    def _load_robustness_metrics(self):
        robustness_args = {'return_aggregate': self.aggregate,
                           'disable_warnings': self.disable_warnings,
                           'display_progressbar': False,
                           }
        # todo fix explanation method parameter for robustness metrics
        self.robustness_metrics = {
            'local_lipschitz_estimate': quantus.LocalLipschitzEstimate(
                nr_samples=10,
                perturb_std=0.2,
                perturb_mean=0.0,
                norm_numerator=quantus.distance_euclidean,
                norm_denominator=quantus.distance_euclidean,
                perturb_func=quantus.gaussian_noise,
                similarity_func=quantus.lipschitz_constant,
                **robustness_args
            ),
            'max_sensitivity': quantus.MaxSensitivity(
                nr_samples=10,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **robustness_args
            ),
            'average_sensitivity': quantus.AvgSensitivity(
                nr_samples=10,
                lower_bound=0.2,
                perturb_func=quantus.uniform_noise,
                similarity_func=quantus.difference,
                **robustness_args
            ),
            'continuity': quantus.Continuity(
                patch_size=56,
                nr_steps=10,
                perturb_baseline="uniform",
                similarity_func=quantus.correlation_spearman,
                **robustness_args
            ),
            'consistency': quantus.Consistency(
                **robustness_args
            ),
            'relative_input_stability': quantus.RelativeInputStability(
                nr_samples=5,
                **robustness_args),
            'relative_output_stability': quantus.RelativeOutputStability(
                nr_samples=5,
                **robustness_args),
            'relative_representation_stability': quantus.RelativeRepresentationStability(
                nr_samples=5,
                **robustness_args),
        }

    def _load_localization_metrics(self):
        localization_args = {'return_aggregate': self.aggregate,
                             'disable_warnings': self.disable_warnings,
                             'display_progressbar': True,
                             }
        self.localization_metrics = {
            'pointing_game': quantus.PointingGame(**localization_args),
            'attribution_localisation': quantus.AttributionLocalisation(**localization_args),
            'top_k_intersection': quantus.TopKIntersection(**localization_args),
            'relevance_rank_accuracy': quantus.RelevanceRankAccuracy(**localization_args),
            'relevance_mass_accuracy': quantus.RelevanceMassAccuracy(**localization_args),
            'auc': quantus.AUC(**localization_args),
        }

    def _load_complexity_metrics(self):

        pass

    def _load_randomization_metrics(self):
        pass

    def _load_axiomatic_metrics(self):
        pass
