import os


class BaseDataset(object):
    def __init__(self):
        pass
    def __check_dir(self):
        raise NotImplementedError()