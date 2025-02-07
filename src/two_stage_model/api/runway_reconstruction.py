import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from io import BytesIO
from typing import *

from utils import load_torch_model

from log import logger

class RunwayReconstruction():
    """
    Reconstruct runway images to low-res FOD-faded images
    """
    def __init__(self, 
                 modelPath: str, 
                 net: Any, 
                 removeGrass: bool,
                 transform: Optional[Type[torchvision.transforms.Compose]]=None, 
                 useGPU: bool=False) -> None:
        """
        Initial variables for RunwayReconstruction class object 
        inputs:
            modelPath: the reconstrution model path
            net: the model structure class
            transform: default to be None. If passed, it will be the transforms that would be posed to input images
            useGPU: whether gpu is available and to be used
        defined variables:
            self.transform: the passed transform or default transform
            self.GPU: whether gpu is available and to be used
            self.model: the loaded model
            self.device: cuda if cuda is available else cpu
        """
        self.transform = transform if transform else torchvision.transforms.Compose(
                        [torchvision.transforms.Grayscale(num_output_channels=1),
                        torchvision.transforms.Resize((256,256)),
                        torchvision.transforms.ToTensor()])
        self.GPU = useGPU
        self.model = load_torch_model(modelPath, net, gpu=self.GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.removeGrass = removeGrass

    def reconstruct(self, corr_ID: str, image: Union[str, BytesIO], fileName: Optional[str] = None) -> Image:
        """
        Reconstruct the input image to a low-res FOD-faded image
        inputs:
            corrId: [a placeholder]
            image_path: the image path
        outputs:
            reconstructedRunwayImage: a reconstructed PIL Image image of thr input runway 
            transformedRunwayImage: a transformed PIL Image image of thr input runway 
        """
        runwayImage = Image.open(image)
        if not isinstance(image, BytesIO):
            imageName = image.split("/")[-1][:-3] + "png"
        else:
            imageName = fileName
        if self.removeGrass:
            try:
                runwayMask = Image.open(f"../FOD_rgb/masks/{imageName}")
                runwayImage = Image.composite(runwayImage, runwayMask, runwayMask.convert("L"))
            except:
                logger.warning("No mask existed for %s", imageName)
        transformedImage = self.transform(runwayImage).to(self.device)
        print(f"Reconstructing: {imageName}")
        output = self.model(torch.unsqueeze(transformedImage, 0))
        reconstructedRunwayImage = F.to_pil_image(output[0])
        transformedRunwayImage = F.to_pil_image(transformedImage)

        return reconstructedRunwayImage, transformedRunwayImage
