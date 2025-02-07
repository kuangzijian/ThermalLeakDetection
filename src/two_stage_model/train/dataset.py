import torchvision.datasets as datasets
import numpy as np
from PIL import Image

class LeakingDataset(datasets.ImageFolder):
    def __init__(self, root, transforms_label, transform=None, target_transform=None, loader=datasets.folder.default_loader, is_valid_file=None):
        super(LeakingDataset, self).__init__(root, transform=transform, target_transform=target_transform, loader=loader, is_valid_file=is_valid_file)
        self.transforms_label = transforms_label

    def __getitem__(self, index):
        path, target = self.samples[index]
        dpath = path.replace("thermal", "diff")
        sample = self.loader(path)
        dsample = self.loader(dpath)
        sample = sample.convert('RGB')
        dsample = dsample.convert('L')
        # dsample = ImageOps.invert(dsample) # comment out if using difference-highlighted maps
        darray = np.array(dsample)
        darray = np.clip((darray + 20).astype(np.uint8), 0, 255)
        dsample = Image.fromarray(darray)
        sample = Image.merge('RGBA', sample.split() + (dsample,))
        transform = self.transforms_label.get(target) # Use the label-specific transform
        if transform is not None:
            sample = transform(sample)
        else:
            # Fallback to the default transform if label-specific transform not found
            if self.transform is not None:
                sample = self.transform(sample)

        return sample, target