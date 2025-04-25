import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Battery prognostic configer')
args = parser.parse_args()
args.mode = ['oxford','tongji','xijiao','hkust']
args.data_path= './datasets/'
args.model_path = {'oxford':'./log/oxford.h5',
                   'tongji':'./log/tongji.h5',
                   'xijiao':'./log/xijiao.h5',
                   'hkust':'./log/hkust.h5'}

args.lookback=42
args.batch_size=64
args.epochs = 200
args.exclude_rate = 1/3
args.cycle_num=0
args.start_index = 20

args.train_folder = '/train_data/'
args.test_folder = '/test_data/'
args.start_voltage={'oxford':3.0,
            'tongji':3.0,
            'xijiao':3.4,
          'hkust':3.5}
args.end_voltage={'oxford':4.19,
            'tongji':4.2,
            'xijiao':4.2,
          'hkust':4.19}

args.voltage={'oxford':np.linspace(3.0,4.19,120),
            'tongji':np.linspace(3.0,4.2,121),
            'xijiao':np.linspace(3.4,4.2,81),
            'hkust':np.linspace(3.5,4.19,70)}