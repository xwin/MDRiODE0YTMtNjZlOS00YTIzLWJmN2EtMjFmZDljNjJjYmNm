'''
Base HistoryModule class for recording and saving training history/curves
'''

import abc


class history:
    '''
    HistoryModule: Abstract Class
    Entries: history_name: name of the history module
             history_description: textual description of the history module
             data: list of dictionary records containing step/epoch metrics
             history_destination_folder_path: destination folder for saving
             history_destination_file_name: destination file name for saving
    '''

    history_name = None
    history_description = None

    data = None

    history_destination_folder_path = None
    history_destination_file_name = None

    def __init__(self, hName=None, hDescription=None):
        self.history_name = hName
        self.history_description = hDescription
        self.data = []

    def record(self, **kwargs):
        '''Append a step/epoch record containing arbitrary metrics'''
        self.data.append(kwargs)

    @abc.abstractmethod
    def save(self):
        return

    @abc.abstractmethod
    def load(self):
        return
