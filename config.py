import torch
# Train parameters
EPOCH = 50
BATCH_SIZE = 16
BETAS = (0.5, 0.999)
DATASET_FRAC = 1.0
TRAIN_DATASET_FRAC = 1.0
VALID_DATASET_FRAC = 0.2
TEST_DATASET_FRAC = 1.0
LEARNING_RATE = 0.0002
SAVE_PER_RATIO = 0.2


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PREFIX = 'pretrain'


# Ditectory path
CHECKPOINT_DIR_PATH = './checkpoint'
CHECKPOINT_PATH = ''            # ex) '20200914_005102_pretrain'
CHECKPOINT_EPOCH = None
CHECKPOINT_STEP = None
LOG_PATH = './logs'
DATASET_PATH = './dataset'
DEFINITION_PATH = './dataset/3DMM_definition/'


# Normalization type and lambda of loss
SHAPE_LOSS_TYPE = 'l2'
SHAPE_LAMBDA = 10
M_LOSS_TYPE = 'l2'
M_LAMBDA = 5

LANDMARK_LOSS_TYPE = 'l2'
LANDMARK_LAMBDA = 0.02

BATCHWISE_WHITE_SHADING_LOSS_TYPE = 'l2'
BATCHWISE_WHITE_SHADING_LAMBDA = 10

RECONSTRUCTION_LOSS_TYPE = 'l1'
RECONSTRUCTION_LAMBDA = 10

TEXTURE_LOSS_TYPE = 'l1'
TEXTURE_LAMBDA = 1
SMOOTHNESS_LOSS_TYPE = 'l2'
SMOOTHNESS_LAMBDA = 1000

SYMMETRY_LOSS_TYPE = 'l1'
SYMMETRY_LAMBDA = 1

CONST_ALBEDO_LOSS_TYPE = 'l1'
CONST_ALBEDO_LAMBDA = 5

CONST_LOCAL_ALBEDO_LOSS_TYPE = 'l2,1'
CONST_LOCAL_ALBEDO_LAMBDA = 10


# 3DMM parameters
VERTEX_NUM = 53215
TRI_NUM = 105840
N = VERTEX_NUM * 3
IMAGE_SIZE = 224
TEXTURE_SIZE = [192, 224]
LANDMARK_NUM = 68
C_DIM = 3
CONST_PIXELS_NUM = 20


# Log parameters
LOSS_LOG_INTERVAL = 20
IMAGE_LOG_INTERVAL = 20
IMAGE_LOG_NUMBER = 8
