'# -*- coding: utf-8 -*-'
#change args.start_index and args.cycle_num to test input sample with different cycles and starting voltages

from module.data_loader import Preprocess
from module.config import args
from module import utils
from tensorflow.keras import models
import tensorflow as tf

print("Is GPU available:",tf.test.is_gpu_available())
mode = args.mode[0] # args.mode = [0:'oxford',1:'tongji',2:'xjtu',3:'hkust']
model_path = args.model_path[mode]
pr = Preprocess(mode)
test_x, input_one_cycle = pr.get_test_data(args.test_folder)
mean,std = pr.get_mean_std()

model_test = models.load_model(model_path, compile=False)
utils.plot_cycle(model_test,test_x,mode,input_one_cycle,mean,std)
