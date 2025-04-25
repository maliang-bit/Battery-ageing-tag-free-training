'# -*- coding: utf-8 -*-'
from module.data_loader import Preprocess
from module.model import my_network
from module.config import args
import tensorflow as tf

print("Is GPU available:",tf.test.is_gpu_available())
mode = args.mode[0] # args.mode = ['oxford','tongji','xijiao','hkust']
model_path = args.model_path[mode]
pr = Preprocess(mode)
train_x,train_y_lower,train_y_upper,y_ic,weight_all = pr.get_train_data(args.train_folder)

mean,std = pr.get_mean_std()
train_x = (train_x - mean) / std #standardisation

my_network(args.batch_size, args.epochs, train_x, train_y_lower, train_y_upper,
           y_ic,weight_all, model_path)