import json
from collections import defaultdict
import itertools
from os.path import join

# dataset params
def get_train_config(json_path):
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config

def get_train_configs(json_path):
    with open(json_path) as json_file:
        config = json.load(json_file)
        # find free hyper-params
        free_hp = []
        free_fields = []
        for field in config:
            if isinstance(config[field], dict): # free hyper-param
                free_fields.append(field)
                free_vs = []
                for v in config[field].values():
                    free_vs.append(v)
                free_hp.append(free_vs)

        if len(free_fields) == 0:
            return [config], ['.']

        # generate all combinations of hp
        free_hp_comb_list = list(itertools.product(*free_hp))

        # write the combination back to dict object
        ret_config_list = []
        ret_rel_path_list = []
        for free_hp_comb in free_hp_comb_list:
            ret_config = dict(config)
            rel_path = []
            # modify the free field with the value in each combination
            for field, v in zip(free_fields, free_hp_comb):
                ret_config[field] = v
                rel_path.append(field+'_'+str(v).replace('[', '').replace(']',''))
            ret_config_list.append(ret_config)
            ret_rel_path_list.append(join(*rel_path))
    return ret_config_list, ret_rel_path_list

