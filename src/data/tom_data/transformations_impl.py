from torchvision import transforms


class TransformationsImpl:
    def __init__(self, transformations_config: dict, transformations: list):
        self.transformations_config = transformations_config
        self.transformations_list = transformations
        self.composed_transforms = None

    def __call__(self, *args, **kwargs):
        return self.get_transforms(compose=True)(*args, **kwargs)

    def get_transforms(self, compose=True):
        if compose:
            if self.composed_transforms is None:
                self.composed_transforms = self.setup_compose()
            return self.composed_transforms
        else:
            return self.transformations_list

    def add_data_transforms(self, means, stds):
        self.transformations_list.append(transforms.Normalize(mean=means, std=stds))

    def setup_compose(self):
        return transforms.Compose(self.transformations_list)
