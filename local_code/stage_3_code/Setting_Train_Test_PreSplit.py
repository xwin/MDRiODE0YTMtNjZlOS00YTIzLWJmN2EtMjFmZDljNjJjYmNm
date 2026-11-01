'''
Concrete SettingModule class for pre-split train and test datasets
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test_PreSplit(setting):
    dataset = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate

    def print_setup_summary(self):
        print('dataset:', self.dataset.dataset_name,
              ', method:', self.method.method_name,
              ', setting:', self.setting_name,
              ', result:', self.result.result_name,
              ', evaluation:', self.evaluate.evaluate_name)

    def load_run_save_evaluate(self):
        # load datasets
        loaded_train_dataset, loaded_test_dataset = self.dataset.load()
        # run MethodModule
        self.method.data = {'train': loaded_train_dataset, 'test': loaded_test_dataset}
        learned_result = self.method.run()

        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), 0.0
