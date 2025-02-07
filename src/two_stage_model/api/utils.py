import torch
from torch import nn
import numpy as np
import random
from PIL import Image
import cv2
import io

def setup_seed(seed):
    """
    Function to set up random seed.
    Inputs:
        seed: the random seed value
    outputs: 
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_torch_model(PATH, net, gpu=True):
    """
    Function to load torch model.
    Inputs:
        PATH: model path
        net: net architecture
        gpu: use GPU or not
    outputs: 
        model: loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    if gpu: model.load_state_dict(torch.load(PATH))
    else: model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess(bytes, size=None):
    """
    Function to preprocess the to-be-predicted images.
    Inputs:
        bytes: Image bytes
    outputs: 
        data: Resized and dimension-expanded data
    """
    img = cv2.imdecode(bytes, cv2.IMREAD_COLOR)
    if size is not None:
        img = cv2.resize(img, size)
    data = np.expand_dims(img,axis=0).astype(np.float32)
    return data

def bytize_PIL(img):
    """
    Function to convert PIL image to byte stream.
    Inputs:
        img: PIL image
    outputs: 
        bytes: bytes
    """
    with io.BytesIO() as buf:
        img.save(buf, format="JPEG")
        bytes = buf.getvalue()
    return bytes

def cv2_2_pil(cv_img):
    if cv_img.shape[-1] == 4:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
    elif cv_img.shape[-1] == 3:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(cv_img)
    return im_pil

def pil_2_cv2(pil_img: Image, gray=False):
    """
    Function to convert PIL image to cv2(numpy) image.
    Inputs:
        img: PIL image
    outputs: 
        result: numpy image
    """
    numpy_img = np.array(pil_img)
    result = numpy_img[:, :].copy()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    if gray:
        result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    return result

def get_filename_from_path(path: str):
    """
    Function to parse file name from path.
    Inputs:
        path: file path
    outputs: 
        result: the file name
    """
    result = path.split("/")[-1]
    return result

def check_folder(path: str):
    """
    Function to check the input path is an image or not.
    Inputs:
        path: to-be-checked path
    outputs: 
        result: a boolean value indicating the result
    """
    result = ("." not in get_filename_from_path(path))
    return result

def normalize(arr):
    """
    Function to normalize array.
    Inputs:
        arr: the input array
    outputs: 
        result: the normalize array
    """
    if len(arr) == 1: return np.array([1.0])
    if len(arr) == 0: return np.array([])
    result = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
    return result

def update_model_params(model):
    for param in model.parameters():
        param.requires_grad=False
        model.classifier = nn.Sequential(
                            nn.Linear(in_features=1792, out_features=625),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(in_features=625, out_features=256),
                            nn.ReLU(),
                            nn.Linear(in_features=256, out_features=2)
                        )
    return model

