import cv2
import numpy as np
import os
from diff import calculate_ssim
import imageio
from net import ResNet, BasicBlock, Bottleneck
from collections import namedtuple
import torch
from api.utils import cv2_2_pil
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets

def init_model(path='tut5-model.pt'):
    if path is None:
        return None
    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet50_config = ResNetConfig(block = Bottleneck,
                               n_blocks = [3, 4, 6, 3],
                               channels = [64, 128, 256, 512])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(resnet50_config, 2).to(device)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
    
    def forward(self, x):
        x = self.features(x)
        return x
    
if __name__ == "__main__":

    pretrained_means = [0.924, 0.924, 0.924]
    pretrained_stds= [0.072, 0.072, 0.072]

    train_transforms = transforms.Compose([
                            transforms.Resize(224),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            #  transforms.Grayscale(num_output_channels = 1),
                            transforms.RandomCrop(224, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means,
                                                    std = pretrained_stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            #  transforms.Grayscale(num_output_channels = 1),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = pretrained_means,
                                                    std = pretrained_stds)
                        ])

    train_dir = '../dataset/LeakAI_v11_diff_patch_cls_data/train'
    val_dir = '../dataset/LeakAI_v11_diff_patch_cls_data/valid'
    test_dir = '../dataset/LeakAI_v11_diff_patch_cls_data/test'

    train_data = datasets.ImageFolder(root = train_dir,
                                    transform = train_transforms)

    valid_data = datasets.ImageFolder(root = val_dir,
                                    transform = test_transforms)

    test_data = datasets.ImageFolder(root = test_dir,
                                    transform = test_transforms)

    model = init_model()

    feature_extractor = FeatureExtractor(model)

    # Assuming your dataset is ready and preprocessed
    data_loader = DataLoader(train_data, batch_size=64, shuffle=False)

    features, labels = [], []

    from tqdm import tqdm
    with torch.no_grad():
        for data, target in tqdm(data_loader):
            output = feature_extractor(data)
            features.extend(output.cpu().numpy())
            labels.extend(target.numpy())

    # Convert lists to NumPy arrays if not already
    import numpy as np
    features = np.array(features).squeeze()
    labels = np.array(labels)

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Split your data
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

    # Initialize and train the SVM
    clf = SVC()  # Feel free to experiment with different kernels
    clf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
