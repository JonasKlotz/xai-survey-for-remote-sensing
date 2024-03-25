import numpy as np

from visualization.ml_plotly.plotly_helpers import plot_square_object_array
from visualization.ml_plotly.plotly_objects import (
    ImageObject,
    AttributionObject,
    SegmentationObject,
)


def test():
    num_classes = 6
    # Use ``default_rng()`` to create a `Generator` and call its methods.
    rng = np.random.default_rng(seed=42)

    # generate random image data with shape (3, 120, 120) and values within 0-1
    random_image_data = rng.random(size=(3, 120, 120))

    # generate random attribution data with shape (1, 120, 120) and values within 0-1
    random_attribution_data = rng.random(size=(1, 120, 120))

    # generate random segmentation data with shape (1, 120, 120) and values beeing 0 beweetn num_classes
    random_segmentation_data = rng.integers(0, num_classes, size=(1, 120, 120))

    # generate Plotly objects
    image = ImageObject(
        random_image_data, dataset_name="caltech101", title="Test Image"
    )
    attribution = AttributionObject(random_attribution_data, title="Test Attribution")
    segmentation = SegmentationObject(
        random_segmentation_data, num_classes=num_classes, title="Test Segmentation"
    )

    # # create a figure with the Plotly objects
    array_to_plot = [[image, attribution], [segmentation, attribution]]
    fig = plot_square_object_array(array_to_plot, title="Test", figsize=(15, 15))

    assert fig is not None

    # create a figure with the Plotly objects


if __name__ == "__main__":
    test()
