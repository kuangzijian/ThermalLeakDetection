import torch
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F
from PIL import Image
from PIL.Image import Image as img
from io import BytesIO
from typing import *
import cv2
import numpy as np
from utils import update_model_params, pil_2_cv2, cv2_2_pil
from diff import calculate_ssim
import config
import timm

class leakingClassifier():
    def __init__(self, 
                 transform: Optional[Type[T.Compose]]=None, 
                 useGPU: bool=False) -> None:
        self.transform = transform if transform else T.Compose(
                        [T.Resize(size=(config.INPUT_SIZE, config.INPUT_SIZE)), 
                             T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        self.labels = ['Normal', 'Leaking']
        self.GPU = useGPU
        self.model = timm.create_model(config.MODEL_NAME, pretrained=True)
        self.model = update_model_params(self.model)
        self.activations = {}
        self.layer_name = 'bn2'
        getattr(self.model, self.layer_name).register_forward_hook(self.get_activation(self.layer_name))
        # self.model = ModifiedEfficientNet()
        # print(self.model)
        self.model.load_state_dict(torch.load(config.MODEL_PATH, map_location='cpu'))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_activation(self, name):
        def hook(model, input, output):
            self.activations[name] = output.detach()
        return hook

    def predict(self, data: Union[img, str, List[img], List[str]]):
        if isinstance(data, List):
            assert len(data) == 2
            refFrame, curFrame = data
            if isinstance(refFrame, img) and isinstance(curFrame, img):
                npRefFrame, npCurFrame = pil_2_cv2(refFrame, gray=True), pil_2_cv2(curFrame, gray=True)
            elif isinstance(refFrame, str) and isinstance(curFrame, str):
                npRefFrame, npCurFrame = cv2.imread(refFrame, cv2.IMREAD_GRAYSCALE), cv2.imread(curFrame, cv2.IMREAD_GRAYSCALE)
            _, diff = calculate_ssim(npRefFrame, npCurFrame)
            diff = cv2.merge([diff, diff, diff]) * 255.0
            npDiff = diff.copy()
            diff = cv2_2_pil(diff.astype("uint8"))
        else:
            diff = Image.open(data).convert('RGB')
        transformedDiff = self.transform(diff).to(self.device)
        output = self.model(transformedDiff.unsqueeze(0))
        predictions = F.softmax(output, dim = 1)
        prediction = self.labels[np.argmax(predictions.detach().numpy(), 1)[0]]

        try:
            return output, prediction, npDiff
        except:
            return output, prediction
    
    def get_intermediate_activations(self):
        # Method to access the activations
        return self.activations
    
# class ModifiedEfficientNet(nn.Module):
#     def __init__(self):
#         super(ModifiedEfficientNet, self).__init__()
#         self.model = timm.create_model(config.MODEL_NAME, pretrained=True)
#         self.model = update_model_params(self.model)

#     def forward(self, x):
#         # x_hidden = self.model.forward_features(x)
#         # x = self.model.forward_head(x_hidden)
#         # return x_hidden, x
#         x = self.model.forward(x)
#         return x

def gen_heatmap_from_activations(acts):
    acts = acts.squeeze().mean(0)
    acts = (acts - torch.min(acts)) / (torch.max(acts) - torch.min(acts))
    resized_map = torch.nn.functional.interpolate(acts.unsqueeze(0).unsqueeze(0),
                                                  size=(480, 640),
                                                  mode='bilinear',
                                                  align_corners=False)
    resized_map_np = (resized_map.cpu().detach().numpy()*255.0).astype("uint8")
    acts = np.squeeze(resized_map_np)
    heatmap = cv2.applyColorMap(acts, cv2.COLORMAP_JET)
    return heatmap

def overlay_heatmap(base, hm):
    return 0.5 * base + 0.5 * hm

    
if __name__ == "__main__":
    path = "../bincls_test_dataset/positive/61850010038013850010038013850010038013850010038013231220175519630435_diff.jpg"
    classifier = leakingClassifier()
    pred = classifier.predict(path)
    activations = classifier.get_intermediate_activations()[classifier.layer_name]
    heatmap = gen_heatmap_from_activations(activations)
    im = cv2.imread(path)
    overlayed_hm = overlay_heatmap(im, heatmap)
    cv2.imwrite("hm.jpg", overlay_heatmap)

