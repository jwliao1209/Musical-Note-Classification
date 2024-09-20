import os


PROJECT_NAME = "DeepMIR_HW0"
RESULT_DIR = 'results'
CHECKPOINT_DIR = 'checkpoints'
MEL_SPEC = "mel_spec"
MEL_DIR = os.path.join(RESULT_DIR, MEL_SPEC)
CONFIG_FILE = "config.json"
CKPT_FILE = "checkpoint.pth"


FFT_WINDOW_SIZE = 2048
HOP_LENGTH = 512

FEATURE_COLUMNS = [
    'mfcc_mean_0',
    'mfcc_mean_1',
    'mfcc_mean_2',
    'mfcc_mean_3',
    'mfcc_mean_4',
    'mfcc_mean_5',
    'mfcc_mean_6',
    'mfcc_mean_7',
    'mfcc_mean_8',
    'mfcc_mean_9',
    'mfcc_mean_10',
    'mfcc_mean_11',
    'mfcc_mean_12',
    'mfcc_std_0',
    'mfcc_std_1',
    'mfcc_std_2',
    'mfcc_std_3',
    'mfcc_std_4',
    'mfcc_std_5',
    'mfcc_std_6',
    'mfcc_std_7',
    'mfcc_std_8',
    'mfcc_std_9',
    'mfcc_std_10',
    'mfcc_std_11',
    'mfcc_std_12',
    'chroma_mean_0',
    'chroma_mean_1',
    'chroma_mean_2',
    'chroma_mean_3',
    'chroma_mean_4',
    'chroma_mean_5',
    'chroma_mean_6',
    'chroma_mean_7',
    'chroma_mean_8',
    'chroma_mean_9',
    'chroma_mean_10',
    'chroma_mean_11',
    'chroma_std_0',
    'chroma_std_1',
    'chroma_std_2',
    'chroma_std_3',
    'chroma_std_4',
    'chroma_std_5',
    'chroma_std_6',
    'chroma_std_7',
    'chroma_std_8',
    'chroma_std_9',
    'chroma_std_10',
    'chroma_std_11',
    'spectral_contrast_mean_0',
    'spectral_contrast_mean_1',
    'spectral_contrast_mean_2',
    'spectral_contrast_mean_3',
    'spectral_contrast_mean_4',
    'spectral_contrast_mean_5',
    'spectral_contrast_mean_6',
    'spectral_contrast_std_0',
    'spectral_contrast_std_1',
    'spectral_contrast_std_2',
    'spectral_contrast_std_3',
    'spectral_contrast_std_4',
    'spectral_contrast_std_5',
    'spectral_contrast_std_6',
    'zcr_mean',
    'zcr_std',
    'spectral_bandwidth_mean',
    'spectral_bandwidth_std',
    'rms_mean',
    'rms_std',
    'tempo'
]
LABEL = 'label'
KNOWN_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_CLASSES = len(KNOWN_CLASSES)
