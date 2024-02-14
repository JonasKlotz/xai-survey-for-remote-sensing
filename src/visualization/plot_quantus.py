import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Plotting specifics.
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

# Plotting configs.
sns.set(font_scale=1.5)
print(plt.style.available)

plt.style.use("seaborn-white")
plt.rcParams["ytick.labelleft"] = True
plt.rcParams["xtick.labelbottom"] = True

include_titles = True
include_legend = True


# Source code: https://github.com/understandable-machine-intelligence-lab/Quantus/blob/main/tutorials/Tutorial_SFB_Spring_School_2023.ipynb


def spyder_plot(num_vars, frame="circle"):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default."""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default."""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, angles=None):
            self.set_thetagrids(angles=np.degrees(theta), labels=labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """Draw. If frame is polygon, make gridlines polygon-shaped."""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )

                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)

    return theta


def plot_spider_graph(
    normalised_rank_df: pd.DataFrame, xai_methods: list, colours_order: list
):
    # Make spyder graph!
    data = [normalised_rank_df.columns.values, (normalised_rank_df.to_numpy())]
    theta = spyder_plot(len(data[0]), frame="polygon")
    _ = data.pop(0)

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection="radar"))
    fig.subplots_adjust(top=0.85, bottom=0.05)
    for i, (d, method) in enumerate(zip(data[0], xai_methods)):
        _ = ax.plot(theta, d, label=method, color=colours_order[i], linewidth=5.0)
        ax.fill(theta, d, alpha=0.15)

    # Set labels.
    if include_titles:
        ax.set_varlabels(
            labels=[
                "Faithfulness",
                "Localisation",
                "\nComplexity",
                "\nRandomisation",
                "Robustness",
            ]
        )
    else:
        ax.set_varlabels(labels=[])

    ax.set_rgrids(np.arange(0, normalised_rank_df.values.max() + 0.5), labels=[])

    # Set a title.
    ax.set_title(
        "Quantus: Summary of Explainer Quantification",
        position=(0.5, 1.1),
        ha="center",
        fontsize=20,
    )

    # Put a legend to the right of the current axis.
    if include_legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    plt.show()


def normalize_df(df: pd.DataFrame):
    """

    To compare the different XAI methods, we normalise the metric scores between 0 and 1 and rank the scores from lowest
    to highest (i.e. the highest rank corresponds to best performance).

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the metric scores for each XAI method.

    Returns
    -------
    df_normalised_rank : pd.DataFrame

    """
    # Take inverse ranking for Robustness, since lower is better.
    df_normalised = df.loc[:, ~df.columns.isin(["Robustness", "Randomisation"])].apply(
        lambda x: x / x.max()
    )
    df_normalised["Robustness"] = df["Robustness"].min() / df["Robustness"].values
    df_normalised["Randomisation"] = (
        df["Randomisation"].min() / df["Randomisation"].values
    )
    df_normalised_rank = df_normalised.rank()
    return df_normalised_rank


def main():
    dummy_data = {
        "Faithfulness": list(np.random.rand(5)),
        "Localisation": list(np.random.rand(5)),
        "Complexity": list(np.random.rand(5)),
        "Randomisation": list(np.random.rand(5)),
        "Robustness": list(np.random.rand(5)),
    }
    df = pd.DataFrame(dummy_data)
    df_norm = normalize_df(df)
    xai_methods = ["LIME", "SHAP", "DeepLift", "GradCAM", "GradCAM++"]
    df_norm.index = xai_methods

    colours_order = [
        "red",
        "darkorange",
        "royalblue",
        "darkgreen",
        "slateblue",
        "purple",
    ]
    plot_spider_graph(df_norm, xai_methods, colours_order)


def plot_method_comp(df_view_ordered):
    # Plot results!
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax = sns.histplot(
        x="Method",
        hue="Rank",
        weights="Percentage",
        multiple="stack",
        data=df_view_ordered,
        shrink=0.6,
        palette="colorblind",
        legend=False,
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_ylabel("Frequency of rank", fontsize=15)
    ax.set_xlabel("")
    ax.set_xticklabels(["A", "B", "C", "D", "SAL", "GS", "IG", "FG"])
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.1),
        ncol=4,
        fancybox=True,
        shadow=False,
        labels=["1st", "2nd", "3rd", "4th"],
    )
    plt.axvline(x=3.5, ymax=0.95, color="black", linestyle="-")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # main()

    # Load data.    #main()
    data = {
        "Method": [
            "A",
            "B",
            "C",
            "D",
            "Saliency",
            "Saliency",
            "Saliency",
            "Saliency",
            "GradientShap",
            "GradientShap",
            "GradientShap",
            "GradientShap",
            "IntegratedGradients",
            "IntegratedGradients",
            "IntegratedGradients",
            "IntegratedGradients",
            "FusionGrad",
            "FusionGrad",
            "FusionGrad",
            "FusionGrad",
        ],
        "Rank": [
            1.0,
            2.0,
            3.0,
            4.0,
            4.0,
            1.0,
            2.0,
            3.0,
            3.0,
            2.0,
            1.0,
            4.0,
            2.0,
            1.0,
            4.0,
            3.0,
            3.0,
            1.0,
            4.0,
            2.0,
        ],
        "Percentage": [
            100,
            100,
            100,
            100,
            41.67,
            33.33,
            16.67,
            8.33,
            41.67,
            33.33,
            16.67,
            8.33,
            33.33,
            25.0,
            25.0,
            16.67,
            33.33,
            25.0,
            25.0,
            16.67,
        ],
    }
    df = pd.DataFrame(data)
    plot_method_comp(df)
