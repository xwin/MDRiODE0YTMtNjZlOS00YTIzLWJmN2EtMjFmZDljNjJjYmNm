#! python3

import sys
import os
from pathlib import Path
import numpy as np
import torch

from local_code.stage_4_code.Dataset_Loader_Reviews import Dataset_Loader_Reviews

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'stage_4_data', 'text_classification')
RESULT_DIR = os.path.join(PROJECT_ROOT, 'result', 'stage_4_result')

# Text classification traning and evaluation script
if __name__ == "__main__":
    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(16)
    #------------------------------------------------------

    dataset_obj = Dataset_Loader_Reviews('Reivews', 'Movie reviews dataset')
    dataset_obj.dataset_source_folder_path = DATA_DIR
    dataset_obj.dataset_source_file_name = 'reviews_encoded.pt'
