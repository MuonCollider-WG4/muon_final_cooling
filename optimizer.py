import sys
sys.path.append("/Users/elenafol/autophase-test/rf-track-2.1")
import RF_Track
import cooling_utils as utils
import numpy as np
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import pybobyqa
import inverse_model
import time
import rft_utils


CELL_N = 2
loss_weights = {"emitt4d": 2, 'transmission':1.5, "espread":5}
print_progress = True
cells_config_file = './FCchannel_reopt_11cells.json'
# Example, cell 2
minp = [0.1, 1, 1, 0.4, -180, 0.5]
maxp = [5.0, 20, 20, 0.8, 90.0, 2.0]
min_Pz = 100
n_part_init = 1000
transmission_min = 0.8


def get_default_param_space(minp, maxp):
    '''
    Default parameter space defined to optimize 
    solenoid length, absorber (LH2) thickness
    and rotation/acceleration 
    '''
    space  = [Real(minp[0], maxp[0], name="ldrift"),
            Integer(minp[1], maxp[1], name='n_cav_rot'),
            Integer(minp[2], maxp[2], name='n_cav_accel'),
            Real(minp[3], maxp[3], name='abs_len'),
            Integer(minp[4], maxp[4], name='rot_phase'),
            Real(minp[5], maxp[5], name="sol_len")
            ]
    return space


@use_named_args(get_default_param_space(minp, maxp))
def get_loss(**opt_params):
    '''
    Input: cell parameters
    Creates a cooling stage, 
    loads the beam and tracks through RF and absorber region.
    Return: loss function is designed to maximize transmission,
    and minimize 6d emittance.
    '''
    RF_Track.rng_set_seed(1245)
    beam_to_track = RF_Track.Bunch6d()
    beam_to_track.load("./repopt_sol/cooled_cell_{}".format(CELL_N-1))
    utils.get_beam_info_6d(beam_to_track)
    opt_params = list(opt_params.values())
    ldrift, n_cav_rot, n_cav_accel, abs_len, rot_phase, sol_len = [float(value) for value in opt_params]

    # intitialize RF system and cooling segment
    cooling_cell_data, rotating_rf_data, accelerating_rf_data = \
        rft_utils.get_cell_data(CELL_N, cells_config_file)
    cooling_cell = rft_utils.CoolingCell(**cooling_cell_data)
    # reset to optimal parameters
    cooling_cell.abs_len = abs_len
    cooling_cell.sol_len = sol_len
    freq, gradient = rft_utils.get_freq_gradient(beam_to_track)
    rotating_rf = rft_utils.RotationRF(freq, gradient, ldrift, n_cav_rot, 0.25)
    accelerating_rf = rft_utils.AccelRF(n_cav_accel, freq, gradient, 0.25)

    # Acceleration
    if CELL_N < 6:
        beam_rotated = rotating_rf.rotate(beam_to_track, beam_to_track.S, cooling_cell.low_bz, rot_phase)
        accelerated_beam = accelerating_rf.accelerate(beam_rotated, beam_rotated.S, cooling_cell.low_bz)
        beam_after_rf = accelerated_beam
    else:
        accelerated_beam = accelerating_rf.accelerate(beam_to_track, beam_to_track.S, cooling_cell.low_bz)
        beam_rotated = rotating_rf.rotate(accelerated_beam, accelerated_beam.S, cooling_cell.low_bz, rot_phase)
        beam_after_rf = beam_rotated

    # if Pz is too low (acceleration parameters not acceptable), increase the loss
    if beam_after_rf.get_info().mean_P < min_Pz:
        return (min_Pz-accelerated_beam.get_info().mean_P) * 1e6
    
    # Cooling
    cooled_beam = cooling_cell.cool_in_cell(beam_after_rf, beam_after_rf.S, use_matching_coils=False)
    transmission = len(cooled_beam.get_phase_space()) / n_part_init*100
    print("==================================")
    if print_progress:
        print("RF req: ", freq, " Gradient: ", gradient)
        print("Beam after RF")
        utils.get_beam_info_6d(beam_after_rf)
        print("Transmission: ", transmission)
        print("Cooled, not Cleaned")
        utils.get_beam_info_6d(cooled_beam)
    if len(cooled_beam.get_phase_space()) > transmission_min*100:
        # utils.plot_phase_space(cooled_beam)
        cooled_beam = utils.clean_robust_cov(cooled_beam)
        # utils.plot_phase_space(cooled_beam)
        # cooled_beam.save("./repopt_sol/cooled_cell_{}_v2".format(CELL_N))
    else:
        print("Too high losses")
        print("Current params: ", repr(opt_params))
        return 1e6*(n_part_init - len(cooled_beam.get_phase_space()))
    espread_cooled_rel = cooled_beam.get_info().sigma_E / cooled_beam.get_info().mean_K * 100 # energy spread in %
    transmission = len(cooled_beam.get_phase_space()) / n_part_init * 100

    if print_progress:
        print("Cooled")
        utils.get_beam_info_6d(cooled_beam)
        print("Transmission incl. cut: ", transmission)
    
    emitt4d = cooled_beam.get_info().emitt_4d
    emittz = cooled_beam.get_info().emitt_z*1e-3
    # extremely high loss if transverse emittance is increasing
    if emitt4d > beam_to_track.get_info().emitt_4d:
        return emitt4d * 1e6
    
    # loss function
    loss = cooled_beam.get_info().emitt_6d \
        + loss_weights['emitt4d'] * emitt4d \
        + loss_weights['espread'] * espread_cooled_rel \
            + loss_weights['transmission']*(100-transmission)

    print("LOSS: ", loss)
    print("Current params: ", repr(opt_params))
    print("==================================")
    print(opt_params)
    # Save data with timestamp in filename
    inverse_model.save_data(f'./opt_data/data_{int(time.time())}.npz', opt_params, beam_to_track, accelerated_beam, cooled_beam, transmission)
    return loss


def optimize_cell(method, init_params, minp, maxp, n_steps):
    '''
    Method: "bobyqa" or "BO"
    init_params:an array of initial parameters
    '''
    n_initial_points = 1
    # check if there are several starting points
    if type(init_params[0]) == list:
        n_initial_points = len(init_params)

    if method == "bobyqa":
        res = pybobyqa.solve(get_loss, init_params, args=(), bounds=(minp, maxp), npt=None,
                rhobeg=0.2, rhoend=1e-5, maxfun=n_steps, nsamples=None,
                user_params=None, objfun_has_noise=False,
                seek_global_minimum=True,
                scaling_within_bounds=True,
                do_logging=True, print_progress=True)
    if method == "BO":
        space = get_default_param_space(minp, maxp)
        res = gbrt_minimize(get_loss,# the function to minimize
                    space,
                    acq_func="EI",      # the acquisition function
                    n_calls = n_steps,         # the number of evaluations of f
                    n_initial_points=n_initial_points,  # the number of random initialization points
                    x0 = init_params,
                    kappa = 2.0,  # exploration vs. exploitation trade-off
                    random_state=5729) 
    return res



# Example: cell 2
beam_params = {'emitt4d_start':221, 'start_P':100, \
               'emitt4d_end':180, 'sigma_t_end':130, 'sigma_E_end':2.0, 
               'transmission':90}

init_params = {'ldrift':0.32, 'n_cav_rot':5, 'n_cav_accel':5, \
                'abs_len':0.47, 'rot_phase':90, 'sol_len':1.5}

# init_params_predicted = inverse_model.get_estimate(\
#                                 beam_params, init_params, 
#                                 train_new_model=False, data_path=None)
n_opt_steps = 1000
data_directory = './opt_data'
res = optimize_cell('BO', list(init_params.values()), minp, maxp, n_opt_steps)
print(" ======= Optimization finished! =========")
opt_params = np.array(res.x) 
print(repr(opt_params))
# test the performance with optimisied parameters
get_loss(opt_params)

# optionally, re-train the surrogate model with new data and new estimated cell parameters
params_predicted = inverse_model.retrain_and_update_params(
                                n_opt_steps, beam_params, init_params, data_directory)
print(params_predicted)
get_loss(params_predicted)



    
    