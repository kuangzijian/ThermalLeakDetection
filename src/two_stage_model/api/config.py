SOURCE_FILES = "../test" # Input file path
TRANSFORMED_FILES = "../transformed" # Folder to save transformed images
DIFFERENTIAL_CRITERION = "ssim" # Differentiation criterion. [ssim, mae, mse]
OUTPUT_DIR = "../out" # Output directory for all runs
OUTPUT_FOLDER = "ssim" # Output folder for current run
RECONSTRUCT = True # Reconstruct runway images or read from local
GRASS_REMOVAL = "pre" # Grass removal mode. [pre: mask before reconstruction, None: do not remove grass, post: mask during differentiating]
FP_REDCUTION = "cal_connected_components" # False positve reduction critetion. [classify: use FOD classify model, cal_connected_components: use segmentation model]

MODEL_PATH = "model_v2.pt"  # Differentiation model path
MODEL_NAME = 'tf_efficientnet_b4_ns'
INPUT_SIZE = 224
FOD_CLASSIFY_MODEL_PATH = 'model/FODClassifier_AugmentedFODnoAE_09_15_2020_v10.h5' # Classification model path
FOD_SEGMENT_MODEL_PATH = "u2netp_bce_itr_28000_train_0.252114_tar_0.023896" # Segmentation model path

LOW_RES_DIM = (256, 256) # Transformation dimension
REGULARIZED_BBX_SIZE = 320 # Fixed bounding box dimension
CLASSIFY_BBX_SIZE = 64 # Classification model input size

AREA_UPPERB = 300 # Upper bound to filter out bounding boxes by area 
AREA_LOWERB = 5 # lower bound to filter out bounding boxes by area 
BORDER_RATIO = 3 # bordre ratio threshold to filter out bounding boxes

CONFIDENCE_THRESHOLD = 0.1 # Confidence scoring threshold

LOG_LEVEL = 'INFO' # Logger severity level
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s' # Logger output format

GPU = False