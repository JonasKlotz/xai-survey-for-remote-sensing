import os
import sys

from torchvision import transforms

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.append(project_root)
print(f"Added {project_root} to path.")


from src.data.tom_data.datamodule import DeepGlobeDataModule  # noqa: E402
from src.data.tom_data.transformations_impl import TransformationsImpl  # noqa: E402


def main(args):
    transforms_train = TransformationsImpl(
        args,
        [
            transforms.ToTensor(),
        ],
    )

    datamodule = DeepGlobeDataModule(args, transforms_train, transforms_train)

    datamodule.setup()
    datamodule.train_dataloader()
    datamodule.val_dataloader()
    datamodule.test_dataloader()

    print("Done")


if __name__ == "__main__":
    dataset_path_cluster = "/media/storagecube/jonasklotz"
    dataset_dir = "ML_DeepGlobe"
    lmdb_path = "/media/storagecube/jonasklotz/deepGlobe/ML_DeepGlobe/patches.lmdb"
    labels_path = "/media/storagecube/jonasklotz/deepGlobe/ML_DeepGlobe/labels.parquet"
    train_csv = (
        "/media/storagecube/jonasklotz/deepGlobe/splits/deepglobe_version1_train.csv"
    )
    test_csv = (
        "/media/storagecube/jonasklotz/deepGlobe/splits/deepglobe_version1_test.csv"
    )

    cfg = {
        "data": {
            "lmdb_path": lmdb_path,
            "labels_path": labels_path,
            "train_csv": train_csv,
            "test_csv": test_csv,
            "temporal_views_path": None,
            "batch_size": 64,
            "num_workers": 8,
            "pin_memory": True,
        }
    }

    main(cfg)
