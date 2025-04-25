# import
import numpy as np
import os
from scipy.io import loadmat
from .config import args

class Preprocess():
    def __init__(self,mode):
        self.mode = mode
        self.path = args.data_path+mode
        self.voltage = args.voltage[mode]
    def get_file_name(self,folder):
        file_names = []
        for file in os.listdir(self.path+folder):
            if file.endswith('.mat'):
                file_names.append(file)
        return file_names

    def get_mean_std(self):
        data_all = self.get_data_all(args.train_folder)
        data_all = data_all.reshape(-1,2)
        mean = np.mean(data_all, axis=0)
        std = np.std(data_all, axis=0)
        return mean,std

    def get_data_all(self,folder):
        file_names = self.get_file_name(folder)
        data_all = []
        for file_name in file_names:
            capacity = loadmat(self.path + folder + file_name)['capacity']
            voltage_all = np.tile(self.voltage,(capacity.shape[0],1))
            voltage_all = np.expand_dims(voltage_all,-1)
            capacity_all = np.expand_dims(capacity,-1)
            data_stack = np.concatenate((capacity_all,voltage_all),-1)
            data_all.append(data_stack)
        data_all = np.concatenate(data_all)
        return data_all

    def get_train_data(self, folder_list):
        dataset = self.get_data_all(folder_list)
        input_all = []
        output_start_all = []
        output_end_all = []
        ic_all = []
        weight_all = []
        for j,data in enumerate(dataset):
            index_ic_peak = np.argmax(np.diff(data[:, 0]))
            index_vol_peak = data[index_ic_peak,1]
            full_length = len(data)
            excluded_length = int(full_length * args.exclude_rate)
            remain_length = full_length - excluded_length
            start_index = np.random.randint(excluded_length+1)
            end_index = start_index + remain_length
            data_new = np.copy(data)[start_index:end_index]
            data_new[:, 0] = data_new[:, 0]-data_new[0, 0]
            weight = np.ones(3,dtype=int)
            if index_vol_peak>data_new[-1, 1] or index_vol_peak<data_new[0, 1]:
                weight[2] = 0
            IC = np.max(np.diff(data_new[:, 0]))
            data_input_cycle = []
            start_output_cycle = []
            end_output_cycle = []
            ic_cycle = []
            weight_cycle = []
            for ii in range(len(data_new) - args.lookback+1):
                data_piece = data_new.copy()[ii:ii + args.lookback]
                middle_point = data_piece[args.lookback//2,0]
                data_piece[:, 0] = data_piece[:, 0] - middle_point
                output_start = data_piece[:1, 0]
                output_end = data_piece[-1:, 0]
                input_data = data_piece[1:-1]
                data_input_cycle.append(input_data)
                start_output_cycle.append(output_start)
                end_output_cycle.append(output_end)
                ic_cycle.append(np.array([IC]))
                weight_cycle.append(weight)
            data_input_cycle = np.array(data_input_cycle)
            start_output_cycle = np.array(start_output_cycle)
            end_output_cycle = np.array(end_output_cycle)
            ic_cycle = np.array(ic_cycle)
            weight_cycle = np.array(weight_cycle)
            input_all.append(data_input_cycle)
            output_start_all.append(start_output_cycle)
            output_end_all.append(end_output_cycle)
            ic_all.append(ic_cycle)
            weight_all.append(weight_cycle)
        input_all = np.concatenate(input_all)
        output_start_all = np.concatenate(output_start_all)
        output_end_all = np.concatenate(output_end_all)
        ic_all = np.concatenate(ic_all)
        weight_all = np.concatenate(weight_all)
        return input_all, output_start_all,output_end_all,ic_all,weight_all

    def get_test_data(self, folder):
        file_name = self.get_file_name(folder)
        data = loadmat(self.path + folder + file_name[0])['capacity']
        voltage_all = np.tile(self.voltage, (data.shape[0], 1))
        voltage_all = np.expand_dims(voltage_all, -1)
        capacity_all = np.expand_dims(data, -1)
        data_stack = np.concatenate((capacity_all, voltage_all), -1)
        input_one_cycle = data_stack[args.cycle_num]
        input_piece = input_one_cycle[args.start_index + 1:args.start_index + args.lookback - 1]
        return input_piece, input_one_cycle