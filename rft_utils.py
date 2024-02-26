import sys
sys.path.append("/Users/elenafol/autophase-test/rf-track-2.1")
import RF_Track
import json
import numpy as np


def setup_beam(emitt4d_micr, emit_z_mm, sigmat_mm, pzinit_mevc, betax, betay, alphax, alphay, npart, decays_on=True):
    Q_nC = 1
    Twiss = RF_Track.Bunch6d_twiss()
    Twiss.emitt_x =  emitt4d_micr   # mm.mrad
    Twiss.emitt_y =  emitt4d_micr   # mm.mrad
    Twiss.beta_x = betax    # m
    Twiss.beta_y =  betay    # m
    Twiss.alpha_x = alphax    # m
    Twiss.alpha_y =  alphay    # m
    Twiss.emitt_z =   emit_z_mm * 1e3
    Twiss.sigma_t =  sigmat_mm # mm
    # Twiss.sigma_pt = 20
    pz_init = pzinit_mevc

    beam_to_track = RF_Track.Bunch6d(RF_Track.muonmass, Q_nC * RF_Track.nC, +1, pz_init, Twiss, npart)
    if decays_on:
        beam_to_track.set_lifetime(RF_Track.muonlifetime)
    return beam_to_track


def get_track_setup():
    track_setup = RF_Track.TrackingOptions()
    track_setup.dt_mm = 2.0 # mm/c
    track_setup.odeint_algorithm = 'rk2' # analytic' # 'rk2', 'rkf45', 'leapfrog', ...
    track_setup.tt_dt_mm = 10.0  # mm/c
    track_setup.cfx_dt_mm =2.0
    return track_setup


def read_json(filename):
    with open(filename, 'r') as file:
        json_data = json.load(file)
        cells =  json_data['cell']
    return cells

def get_freq_gradient(beam_to_track):
    freq = ((299792458.0 * 1e3) / (beam_to_track.get_info().sigma_t*20))*1e-6 # MHz 
    gradient = 1.88 * np.sqrt(freq)
    return freq, gradient


def get_cell_data(cell_n, config_file_path):
    cells = read_json(config_file_path)
    json_data = cells[cell_n]

    cooling_keys = ['cell_n', 'abs_len', 'entr_coil_bz', 'entr_coil_r', 'entr_coil_offset', 
                        'exit_coil_bz', 'exit_coil_r', 'exit_coil_offset', 'sol_len', 'low_bz_cool']
    rotating_rf_keys = ['freq_accel', 'grad_accel', 'drift_len', "nrot",  "cell_len", "phase_rot", "low_bz_rf"]
    accelerating_rf_keys = ["naccel", 'freq_accel', 'grad_accel',"cell_len", "low_bz_rf"]
    cooling_cell_data = {key: json_data[key] for key in cooling_keys}
    rotating_rf_data = {key: json_data[key] for key in rotating_rf_keys}
    accelerating_rf_data = {key: json_data[key] for key in accelerating_rf_keys}
    return cooling_cell_data, rotating_rf_data, accelerating_rf_data


class CoolingCell:
    def __init__(self, cell_n, abs_len, entr_coil_bz, entr_coil_r, entr_coil_offset,
                 exit_coil_bz, exit_coil_r, exit_coil_offset, sol_len, low_bz_cool):
        self.cell_n = cell_n
        self.abs_len = abs_len
        self.entr_coil_bz = entr_coil_bz
        self.entr_coil_r = entr_coil_r
        self.entr_coil_offset = entr_coil_offset
        self.exit_coil_bz = exit_coil_bz
        self.exit_coil_r = exit_coil_r
        self.exit_coil_offset = exit_coil_offset
        self.sol_len = sol_len
        self.low_bz = low_bz_cool


    def cool_in_cell(self, beam, start_cell, use_matching_coils):
        V = RF_Track.Volume()
        V.set_s0(start_cell)
        V.set_static_Bfield(0, 0.0, self.low_bz)
        cell_center = start_cell + 2.0
        absorber = RF_Track.Absorber(self.abs_len, 'liquid_hydrogen')
        hf_solenoid = RF_Track.Solenoid(self.sol_len, 40, 0.16)
        V.add(hf_solenoid, 0, 0, cell_center, 'center')
        V.add(absorber, 0, 0, cell_center, 'center')
        if use_matching_coils:
            C_start = RF_Track.Coil(0.5, self.entr_coil_bz, self.entr_coil_r)
            C_end = RF_Track.Coil(0.5, self.exit_coil_bz, self.exit_coil_r)
            V.add(C_start, 0, 0, cell_center-(self.entr_coil_offset), 'exit')
            V.add(C_end, 0, 0, cell_center+(self.exit_coil_offset), 'entrance')
        V.set_s1(start_cell + 4.0)
        track_setup = get_track_setup()
        beam_after_cell = V.track(beam, track_setup)
        return beam_after_cell  

 
class AccelRF():
    def __init__(self, n_cav_rf, freq_accel, grad_accel, cell_len):
        self.freq_rot = float(freq_accel*1e6) # Hz
        self.gradient = np.array([grad_accel*1e6]) # V/m
        self.nrot = int(n_cav_rf)
        self.cell_len = float(cell_len)

    def accelerate(self, beam, start_volume, bz):
        start_rf_rot = start_volume
        on_crest = -1
        n_cells = 1
        V = RF_Track.Volume()
        V.set_s0(start_volume)
        V.set_static_Bfield(0, 0.0, bz)
        for i in range(self.nrot):
            SW = RF_Track.SW_Structure(self.gradient, self.freq_rot, self.cell_len, on_crest*n_cells)
            V.add(SW, 0, 0, start_rf_rot, 'entrance')
            start_rf_rot = start_rf_rot + SW.get_length() 
        track_setup = get_track_setup()
        beam_after_acceleration = V.track(beam, track_setup)
        return beam_after_acceleration
    

class RotationRF:
    def __init__(self, freq_rot, grad_rot, drift_len, nrot, cell_len):
        self.freq_rot = float(freq_rot*1e6) # Hz
        self.gradient = np.array([grad_rot*1e6]) # V/m
        self.drift_len = drift_len
        self.nrot = int(nrot)
        self.cell_len = float(cell_len)

    def rotate(self, beam, start_volume, bz, rot_phase):
        start_rf_rot = start_volume + float(self.drift_len)
        on_crest = -1
        n_cells = 1
        V = RF_Track.Volume()
        V.set_s0(start_volume)
        V.set_static_Bfield(0, 0.0, bz)
        for i in range(self.nrot):
            SW = RF_Track.SW_Structure(self.gradient, self.freq_rot, self.cell_len, on_crest*n_cells)
            SW.set_phid(float(rot_phase))
            SW.set_t0(0.0) # mm/c
            V.add(SW, 0, 0, start_rf_rot, 'entrance')
            start_rf_rot = start_rf_rot + SW.get_length() 
        track_setup = get_track_setup()
        beam_after_rotation = V.track(beam, track_setup)
        return beam_after_rotation