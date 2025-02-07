from collections import namedtuple
import torchvision.transforms as transforms
from resnet import BasicBlock, Bottleneck

class Config:
    SEED = 1234
    pretrained_size = 224
    means = [0.669, 0.342, 0.479, 0.924]
    stds= [0.215, 0.212, 0.132, 0.072]
    BATCH_SIZE = 8
    n_images_visualize = 25
    start_idx_visualize = 2000
    init_lr = 1e-4
    EPOCHS = 30
    save_dir = "models/"
    model_name = "Leakai_mc_v16.pt"

    train_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means,
                                                    std = stds)
                        ])

    train_transforms_aug = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.RandomRotation(5),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
                            transforms.RandomCrop(pretrained_size, padding = 10),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means,
                                                    std = stds)
                        ])

    test_transforms = transforms.Compose([
                            transforms.Resize(pretrained_size),
                            transforms.CenterCrop(pretrained_size),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = means,
                                                    std = stds)
                        ])

    transforms_label_train = {
        0: train_transforms_aug,
        1: train_transforms,
    }

    transforms_label_test = {
        0: test_transforms,
        1: test_transforms,
    }

    # train_dir = '../../dataset/LeakAI_train_data_20240725/thermal/train'
    # val_dir = '../../dataset/LeakAI_train_data_20240725/thermal/valid'
    # test_dir = '../../dataset/LeakAI_train_data_20240725/thermal/test'

    train_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\combined_data_0808\thermal\train"
    val_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\combined_data_0808\thermal\valid"
    test_dir = r"C:\Users\Lucas\LeakAIEngine\dataset\combined_data_0808\thermal\test"

    ResNetConfig = namedtuple('ResNetConfig', ['block', 'n_blocks', 'channels'])
    resnet18_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [2,2,2,2],
                                channels = [64, 128, 256, 512])

    resnet34_config = ResNetConfig(block = BasicBlock,
                                n_blocks = [3,4,6,3],
                                channels = [64, 128, 256, 512])

    resnet50_config = ResNetConfig(block = Bottleneck,
                                n_blocks = [3, 4, 6, 3],
                                channels = [64, 128, 256, 512])

    resnet101_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 4, 23, 3],
                                    channels = [64, 128, 256, 512])

    resnet152_config = ResNetConfig(block = Bottleneck,
                                    n_blocks = [3, 8, 36, 3],
                                    channels = [64, 128, 256, 512])