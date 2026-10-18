'''
Concrete SettingModule class for pre-split train and test datasets
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test_PreSplit(setting):
    train_dataset = None
    test_dataset = None

    def prepare(self, sTrainDataset, sTestDataset, sMethod, sResult, sEvaluate):
        self.train_dataset = sTrainDataset
        self.test_dataset = sTestDataset
        # Maintain backwards compatibility with base class attributes
        self.dataset = sTrainDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('train dataset:', self.train_dataset.dataset_name,
              ', test dataset:', self.test_dataset.dataset_name,
              ', method:', self.method.method_name,
              ', setting:', self.setting_name,
              ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):
        # load datasets
        loaded_train_data = self.train_dataset.load()
        loaded_test_data = self.test_dataset.load()

        X_train, y_train = loaded_train_data['X'], loaded_train_data['y']
        X_test, y_test = loaded_test_data['X'], loaded_test_data['y']

        # run MethodModule
        self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), 0.0
