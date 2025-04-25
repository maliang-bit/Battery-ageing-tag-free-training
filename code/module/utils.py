import numpy as np
import matplotlib.pyplot as plt
from module.config import args
from sklearn.metrics import mean_squared_error

def prediction_one_cycle(model,test_window,mode,mean,std):
    real_pieces = np.copy(test_window)
    test_x = np.copy(test_window)
    test_x[:,0] = test_x[:,0]-test_x[0, 0]
    outs_lower = np.copy(test_x)
    outs_upper = np.copy(test_x)
    voltage_lower = np.copy(test_x[:1, 1:])
    voltage_upper = np.copy(test_x[-1:, 1:])
    prediction_all = test_x[:, 0].tolist()
    voltage_all = test_x[:, 1].tolist()
    while 1:
        input_lower = outs_lower.copy()
        input_upper = outs_upper.copy()
        middle_lower = input_lower.copy()[(args.lookback - 2) // 2, 0]
        middle_upper = input_upper.copy()[(args.lookback - 2) // 2, 0]
        input_lower[:, 0] = input_lower[:, 0] - middle_lower
        input_upper[:, 0] = input_upper[:, 0] - middle_upper
        input_lower = (input_lower - mean) / std
        input_upper = (input_upper - mean) / std
        input_lower = input_lower.reshape((1, -1, 2))
        input_upper = input_upper.reshape((1, -1, 2))
        prediction_lower, _,_ = model.predict(input_lower, verbose=False)
        _, prediction_upper,_ = model.predict(input_upper, verbose=False)
        capacity_lower = prediction_lower + middle_lower
        capacity_upper = prediction_upper + middle_upper
        voltage_lower -= 0.01
        voltage_upper += 0.01
        new_input_lower = np.concatenate((capacity_lower, voltage_lower), -1)
        new_input_upper = np.concatenate((capacity_upper, voltage_upper), -1)
        outs_lower = np.concatenate((new_input_lower, outs_lower[:-1]), 0)
        outs_upper = np.concatenate((outs_upper[1:], new_input_upper), 0)
        if voltage_upper.flatten()[0] < args.end_voltage[mode]+0.005:
            prediction_all.append(capacity_upper.flatten()[0])
            voltage_all.append(voltage_upper.flatten()[0])
        if voltage_lower.flatten()[0] > args.start_voltage[mode] - 0.005:
            prediction_all.insert(0, capacity_lower.flatten()[0])
            voltage_all.insert(0, voltage_lower.flatten()[0])
        if (voltage_upper.flatten()[0] > args.end_voltage[mode] and
                voltage_lower.flatten()[0] < args.start_voltage[mode]):
            break
    difference = 0 - prediction_all[0]
    prediction_all = prediction_all+difference
    return real_pieces, np.array(prediction_all),voltage_all

def plot_cycle(model, test_x, mode,full_data,mean,std):
    plt.rc('font', family='Arial')
    start_data, predicted_capacity, predicted_voltage = prediction_one_cycle(model, test_x,  mode,mean,std)
    plt.figure(figsize=(8, 4), dpi=200)
    input_voltage = start_data[:, 1]
    real_voltage = full_data[:, 1]
    input_capacity = start_data[:, 0]/ 1000
    real_capacity = full_data[:, 0]/ 1000
    predicted_capacity = predicted_capacity/ 1000
    plt.plot( input_capacity,input_voltage, label='Input piece', lw=5)
    plt.scatter( predicted_capacity,predicted_voltage, label='Predicted capacity', marker='o',
                s=30, c='none', edgecolors='r', lw=1)
    plt.plot( real_capacity,real_voltage, label='Observed capacity', ls='--', color='black')
    plt.xlabel('Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.savefig(f'./figure/{mode}.tif')