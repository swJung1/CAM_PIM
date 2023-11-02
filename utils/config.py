import os
import sys
import yaml


# Slimmable networks
# https://github.com/JiahuiYu/slimmable_networks/blob/5dc14d0357ccfc596d706281acdc8a5b0b66c6d6/utils/config.py

FLAGS = None
Complexity = None

class AttrDict(dict):
    """Dict as attribute trick."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        for key in self.__dict__:
            value = self.__dict__[key]
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, list):
                if isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value


class Config(AttrDict):
    """Config with yaml file.
    This class is used to config model hyper-parameters, global constants, and
    other settings with yaml file. All settings in yaml file will be
    automatically logged into file.
    Args:
        filename(str): File name.
    Examples:
        yaml file ``model.yml``::
            NAME: 'neuralgym'
            ALPHA: 1.0
            DATASET: '/mnt/data/imagenet'
        Usage in .py:
        >>> from neuralgym import Config
        >>> config = Config('model.yml')
        >>> print(config.NAME)
            neuralgym
        >>> print(config.ALPHA)
            1.0
        >>> print(config.DATASET)
            /mnt/data/imagenet
    """

    def __init__(self, filename=None):
        assert os.path.exists(filename), 'File {} not exist.'.format(filename)
        try:
            with open(filename, 'r') as f:
                cfg_dict = yaml.load(f, yaml.FullLoader)
        except EnvironmentError:
            print('Please check the file with name of "%s"', filename)
        super(Config, self).__init__(cfg_dict)
        
        
def app():
    """Load app via stdin from subprocess"""
    global FLAGS
    global Complexity
    global yaml_filename
    
    if FLAGS is None:
        job_yaml_file = None
        for arg in sys.argv:
            # Config file directory: config
            if arg.startswith('yml/'):
                job_yaml_file = arg
        
        # default yml file
        # if job_yaml_file is None:
        #     job_yaml_file = 'yml/default.yml'
            
        FLAGS = Config(job_yaml_file)
        # Complexity = Config('yml/complexity.yml')
        
        # ./yml/resnet20_mixnet.yml (remove yml)
        yaml_filename = job_yaml_file[4:-4]         
        
app()


