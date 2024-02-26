import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor


def save_data(sample_path, opt_params, beam_to_track, accelerated_beam, cooled_beam, transmission):
    ldrift, n_cav_rot, n_cav_accel, abs_len, rot_phase, sol_len = [float(value) for value in opt_params]
    freq = ((299792458.0 * 1e3) / (beam_to_track.get_info().sigma_t*20))*1e-6 # MHz 
    gradient = 1.88 * np.sqrt(freq)
    np.savez(sample_path, ldrift=ldrift, n_cav_rot=n_cav_rot, n_cav_accel=n_cav_accel, \
             abs_len=abs_len, rot_phase=rot_phase, sol_len=sol_len, freq=freq, gradient=gradient, \
                emitt4d_start=beam_to_track.get_info().emitt_4d, 
                emittz_start=beam_to_track.get_info().emitt_z*1e-3, 
                sigma_t_start=beam_to_track.get_info().sigma_t, 
                sigma_E_start=beam_to_track.get_info().sigma_E, 
                start_P=beam_to_track.get_info().mean_P,
                sigma_E_rf=accelerated_beam.get_info().sigma_E,
                sigma_t_rf=accelerated_beam.get_info().sigma_E,
                mean_P_rf = accelerated_beam.get_info().mean_P, 
                mean_P_end = cooled_beam.get_info().mean_P,
                emitt4d_end=cooled_beam.get_info().emitt_4d,
                emittz_end=cooled_beam.get_info().emitt_z*1e-3, 
                sigma_t_end=cooled_beam.get_info().sigma_t, 
                sigma_E_end=cooled_beam.get_info().sigma_E, transmission=transmission)
    
def collect_data(data_directory, beam_params, cell_params):
    # Create lists to store the data
    all_inputs = []
    all_outputs = []

    # Iterate over each file in the directory
    for filename in os.listdir(data_directory):
        if filename.endswith('.npz'):   
            # Load data from the file
            filepath = os.path.join(data_directory, filename)
            data = np.load(filepath)
            if 'start_P' in data.keys():
                sample_input= []
                sample_output = []
                # Append data to observables sample
                for key in beam_params:
                    sample_input.append(data[key])
                # Append data to parameters sample
                for key in cell_params:
                    sample_output.append(data[key])
                # Append parameters and observables to the lists
                all_inputs.append(sample_input)
                all_outputs.append(sample_output)

    # Convert the lists to numpy arrays
    all_inputs = np.array(all_inputs)
    all_outputs = np.array(all_outputs)
    return all_inputs, all_outputs


def get_estimate(beam_params, cell_params, train_new_model, data_directory):
    model = XGBRegressor(booster='dart')
    if train_new_model and data_directory is not None:
        all_inputs, all_outputs = collect_data(data_directory, beam_params.keys(), cell_params.keys())
        X_train, X_test, Y_train, Y_test = train_test_split(
            all_inputs, all_outputs, test_size=0.2, random_state=98)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        r2_val = r2_score(Y_test, y_pred)
        print("R^2 Score on Validation Set:", r2_val)
    else:
        model.load_model('SM_predict_cell-fullData')
        predicted_cell = model.predict(np.array(beam_params.values()).reshape(1, -1))[0]
    return predicted_cell


def retrain_and_update_params(n_last_sampples, beam_params, cell_params, data_directory):
    model = XGBRegressor(booster='dart')
    all_inputs, all_outputs = collect_data(data_directory, beam_params.keys(), cell_params.keys())
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_inputs[-n_last_sampples:], all_outputs[-n_last_sampples:], test_size=0.2, random_state=98)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    r2_val = r2_score(Y_test, y_pred)
    print("R^2 Score on Validation Set:", r2_val)
    predicted_cell = model.predict(np.array(list(beam_params.values())).reshape(1, -1))[0]
    return predicted_cell


def train_model(data_directory, beam_params_names, cell_params_names, model_path, nlast_samples):
    model = XGBRegressor(booster='dart')
    all_inputs, all_outputs = collect_data(data_directory, beam_params_names, cell_params_names)
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_inputs[:-nlast_samples], all_outputs[:-nlast_samples], test_size=0.2, random_state=98)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    r2_val = r2_score(Y_test, y_pred)
    print("R^2 Score on Validation Set:", r2_val)
    model.save_model(model_path)
    return model

# model = train_model('./opt_data', ['emitt4d_start', 'start_P', 'emitt4d_end', 'sigma_t_end','sigma_E_end', 'transmission'],
#                     ['ldrift', 'n_cav_rot', 'n_cav_accel', \
#                 'abs_len', 'rot_phase', 'sol_len'],
#                 './mdl_last_cell', 400)
# predicted_cell = model.predict(np.array([48, 59, 40, 2000, 2.0, 43]).reshape(1, -1))[0]
# print(repr(np.array(predicted_cell)))

