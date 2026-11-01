from local_code.stage_3_code.Dataset_Loader_Pickle import Dataset_Loader_Pickle
from local_code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.History_Saver import History_Saver
from local_code.stage_3_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_3_code.Evaluate_Accuracy import (
    Evaluate_Accuracy,
    Evaluate_Precision,
    Evaluate_Recall,
    Evaluate_F1,
    Evaluate_All
)
import numpy as np
import torch

#---- ORL model training and evaluation script ----
if 1:

    # study can be None, single_layer, dual_layer, no_activation
    study = None
    # possible names: accuracy, precision, recall, f1, all
    evaluator_type = 'all'
    # possible names: Adam, SGD, RMSprop
    optimizer_type = 'Adam'

    #---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    torch.set_num_threads(16)
    #------------------------------------------------------

    # ---- objection initialization setction ---------------
    orl_dataset_obj = Dataset_Loader_Pickle('orl', 'ORL Dataset')
    orl_dataset_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    orl_dataset_obj.dataset_source_file_name = 'ORL'
    orl_dataset_obj.channel_index = 0 # R channel
    orl_dataset_obj.label_offset = 1 # 
    orl_dataset_obj.augment = True

   

    if False:
        train_ds, test_ds = orl_dataset_obj.load()
        x, y = train_ds.tensors
        print(len(train_ds), len(test_ds))             # 360 40
        print(x.shape, x.dtype, y.dtype)               # [360, 1, 112, 92] float32 int64
        print(x.mean().item(), x.std().item())         # ~0.0, ~1.0
        print(y.min().item(), y.max().item())          # 0 39
        print(test_ds.tensors[0].mean().item())        # close to 0, not exactly 0
        exit(0)

    history_obj = History_Saver('training history', '')
    history_obj.history_destination_folder_path = '../../result/stage_3_result/'
    history_obj.history_destination_file_name = 'train_history_orl.csv'


    method_obj = Method_CNN_ORL('CNN', 'orl model')

    method_obj.history = history_obj
    method_obj.optimizer_name = optimizer_type
    # we need to increase learning rate for SGD optimier
    if method_obj.optimizer_name == 'SGD':
        method_obj.learning_rate = 1e-2
    #method_obj.max_epoch = 1

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/ORL_'
    result_obj.result_destination_file_name = 'prediction_result'
    result_obj.fold_count = "1"
    setting_obj = Setting_Train_Test_PreSplit('train test pre split', '')


    if evaluator_type == 'accuracy':
        evaluate_obj = Evaluate_Accuracy('accuracy', '')
    elif evaluator_type == 'precision':
        evaluate_obj = Evaluate_Precision('precision', '')
    elif evaluator_type == 'recall':
        evaluate_obj = Evaluate_Recall('recall', '')
    elif evaluator_type == 'f1':
        evaluate_obj = Evaluate_F1('f1', '')
    else:
        evaluate_obj = Evaluate_All('All metrics', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(orl_dataset_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
   