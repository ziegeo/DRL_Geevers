# -*- coding: utf-8 -*-
"""
Created on Tue May 12 11:05:47 2020

@author: KevinG
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:02:37 2020

@author: LotteH
"""

import numpy as np
import os
import json
import atexit
import datetime
import torch

from support_functions_Lotte import statistics_scalar

class Logger():
    '''
    Initialize a logger object. Args:
        output_dir (string) - the directory where the results will be saved. Default = /tmp/experiments/datetime
        output_fname (string) - name of the file that contains metrics logged throughout the training run. Default = 'progress.txt'
        exp_name (string): Experiment name (if you run the same parameters but just different seeds, keep them the same name!)
    '''
    def __init__(self, output_dir = None, output_fname = 'progress.txt', exp_name = None):
        if output_dir != None: 
            self.output_dir = os.path.join(output_dir.datetime.now().strftime("%Y%m%d-%H%M%S"), exp_name)
        else: 
            self.output_dir = os.path.join("tmp/experiments/", exp_name)
        os.makedirs(self.output_dir)
        self.output_file = open(os.path.join(self.output_dir, output_fname), 'w')
        atexit.register(self.output_file.close)
        print("Logging data to {}".format(self.output_file.name))
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.epoch_dict = dict()
    
    def log(self, msg):
        ''' print file in the console output '''
        print(msg)
    
    def log_tabular(self, key, val):
        '''
        Log a value of a diagnostic. Call this only once for each diagnostic quantity, each iteration.
        After using log_tabular, make sure to call 'dump_tabular' to write them out to the file and console output,
        otherwise they are not saved anywhere.
        '''
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key {} that you didn't include in the \
                                                first iteration".format(key)
        assert key not in self.log_current_row, "You already set {} this iteration. Maybe you forgot to call \
                                                    dump_tabular()".format(key)
        self.log_current_row[key] = val
    
    def save_config(self, config):
        '''
        Log an experiment configuration.
        Load a dictionary with the relevant parameters
        '''
        with open(os.path.join(self.output_dir, "config.txt"), 'w') as out:
            json.dump(config, out, separators = (',', ': '), indent = 2, sort_keys = True)
    
    def dump_tabular(self):
        '''
        Write diagnostics of current iteration, both to console output and the output file
        '''
        vals = []
#        key_lens = [len(key) for key in self.log_headers]
#        max_key_len = max(15, max(key_lens))
#        keystr = '%' + '%d'%max_key_len
#        fmt = '| ' + keystr + 's | %15s |'
#        n_slashes = 22 + max_key_len
#        print("-"*n_slashes)
        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
#            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
#            print(fmt%(key, valstr))
            vals.append(val)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False
        
    def store(self, **kwargs):
        '''save something in epoch_logger current state'''
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)
    
    def log_epoch_tabular(self, key, val=None, with_min_max=False, average_only=False):
        if val is not None:
            self.log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = statistics_scalar(vals, with_min_max = with_min_max)
            self.log_tabular(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                self.log_tabular('Std' + key, stats[1])
            if with_min_max:
                self.log_tabular('Max' + key, stats[3])
                self.log_tabular('Min' + key, stats[2])
        self.epoch_dict[key] = []
        if val == None:
            return stats
    
    def get_stats(self, key):
        v = self.epoch_dict[key]
        print(v)
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
        return statistics_scalar(vals)
    
    def save_state(self, model, pi_optimizer, vf_optimizer, name, itr = None):
        ''' 
        Saves the state_dict of an experiment. Saves the pytorch model (or models). 
        If you want to overwrite the weights you previously had, keep itr = None, otherwise increment itr.
        '''
        fpath = 'pyt_network_save'
        fpath = os.path.join(self.output_dir, fpath)
        fname = name + ('%d'%itr if itr is not None else '') + '.pt'
        fname = os.path.join(fpath, fname)
        os.makedirs(fpath, exist_ok = True)
        torch.save({'model': model.state_dict(), 
                    'pi_optimizer': pi_optimizer.state_dict(),
                    'vf_optimizer': vf_optimizer.state_dict()}, fname)