from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Method_MLP import (
    Method_MLP, 
    Method_MLP_Single_Layer, 
    Method_MLP_Dual_Layer, 
    Method_MLP_No_Activation
)
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.History_Saver import History_Saver
from local_code.stage_2_code.Setting_Train_Test_PreSplit import Setting_Train_Test_PreSplit
from local_code.stage_2_code.Evaluate_Accuracy import (
    Evaluate_Accuracy,
    Evaluate_Precision,
    Evaluate_Recall,
    Evaluate_F1,
    Evaluate_All
)
import numpy as np
import torch

#---- Multi-Layer Perceptron script ----
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
    test_data_obj = Dataset_Loader('mnist', 'mnist Testing data')
    test_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    test_data_obj.dataset_source_file_name = 'test.csv'
  
    train_data_obj = Dataset_Loader('mnist', 'mnist Traning data')
    train_data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    train_data_obj.dataset_source_file_name = 'train.csv'

    history_obj = History_Saver('training history', '')
    history_obj.history_destination_folder_path = '../../result/stage_2_result/'
    history_obj.history_destination_file_name = 'train_history_mlp.csv'

    if study == 'single_layer':
        method_obj = Method_MLP_Single_Layer('single-layer perceptron', '')
    elif study == 'dual_layer':
        method_obj = Method_MLP_Dual_Layer('dual-layer perceptron', '')
    elif study == 'no_activation':
        method_obj = Method_MLP_No_Activation('no activation', '')
    else:
        print('No studies, running multi-layer perceptron.')
        method_obj = Method_MLP('multi-layer perceptron', '')
    method_obj.history = history_obj
    method_obj.optimizer_name = optimizer_type
    # we need to increase learning rate for SGD optimier
    if method_obj.optimizer_name == 'SGD':
        method_obj.learning_rate = 1e-2

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
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
    setting_obj.prepare(train_data_obj, test_data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------
    

    