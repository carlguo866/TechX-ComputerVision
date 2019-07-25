from pathlib import Path
import glob
import numpy as np
import os.path as osp

import utils

class Dataset(object):
    def __init__(self, dataset_root):
        self.root = dataset_root
        self.train = None
        self.test = None
        self.classes = [] # string list
        self.num_classes = None

    def load_dataset(self):
        print('loading data from: {}'.format(self.root))
        if not osp.isdir(self.root):
            raise ValueError('Dataset root directory supplied not found.')
        self.train = self.load_subset(root=self.root,subset_type='train')
        self.test = self.load_subset(root=self.root,subset_type='test')

        # update dataset meta
        self.num_classes = len(self.train)
        self.class_ids = list(range(len(self.classes)))
        self.class_name2id = dict(zip(self.classes, self.class_ids))
        self.class_id2name = dict(zip(self.class_ids, self.classes))
        
        print('data successfully loaded with stats:')
        utils.print_data_stats(self.train,title='train')
        utils.print_data_stats(self.test,title='test')
        
    def load_subset(self,root,subset_type):
        lib = {}
        class_dirs = glob.glob(str(self.root/subset_type/'*'))
        for c in class_dirs:
            c = Path(c)
            class_name = c.stem
            self.classes.append(class_name)
            image_list = glob.glob(str(c/'*.jpg'))
            lib[class_name] = image_list
            assert isinstance(image_list,list)
        return lib
