
import json
import sys, os
sys.path.append("/Users/elenafol/rf_track_scattering_test/rf-track-2.1")
import RF_Track
import cooling_utils as utils
import numpy as np
from sklearn.covariance import EllipticEnvelope, MinCovDet


def read_json(filename):
    with open(filename, 'r') as file:
        json_data = json.load(file)
        cells =  json_data['cell']
    return cells


def cut_distribution(beam, cell_n):
    contamination = 0.01
    cleaned_beam = utils.clean_6dphasespace(beam, contamination)
    if cell_n > 7:
        cleaned_beam =  utils.clean_6dphasespace(cleaned_beam, 0.01)
        cleaned_beam = utils.sigma_cut(cleaned_beam, 3)
    return cleaned_beam


def get_track_setup():
    track_setup = RF_Track.TrackingOptions()
    track_setup.dt_mm = 1.0 # mm/c
    track_setup.odeint_algorithm = 'analytic' # analytic' # 'rk2', 'rkf45', 'leapfrog', ...
    track_setup.tt_dt_mm = 10.0  # mm/c, track the emittance every tt_dt_mm steps
    track_setup.cfx_dt_mm = 2.0
    return track_setup


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
    pz_init = pzinit_mevc

    beam_to_track = RF_Track.Bunch6d(RF_Track.muonmass, Q_nC * RF_Track.nC, +1, pz_init, Twiss, npart)
    if decays_on:
        beam_to_track.set_lifetime(RF_Track.muonlifetime)
    return beam_to_track


class CoolingCell:
    def __init__(self, cell_n, abs_len, entr_coil_bz, entr_coil_r, entr_coil_offset,
                 exit_coil_bz, exit_coil_r, exit_coil_offset, sol_len, low_bz):
        self.cell_n = cell_n
        self.abs_len = abs_len
        self.entr_coil_bz = entr_coil_bz
        self.entr_coil_r = entr_coil_r
        self.entr_coil_offset = entr_coil_offset
        self.exit_coil_bz = exit_coil_bz
        self.exit_coil_r = exit_coil_r
        self.exit_coil_offset = exit_coil_offset
        self.sol_len = sol_len
        self.low_bz = low_bz
    
    def cool_in_cell(self, beam, start_cell, cut=True):
        V = RF_Track.Volume()
        V.set_s0(start_cell)
        V.set_static_Bfield(0, 0.0, self.low_bz)
        cell_center = start_cell + 2.0
        absorber = RF_Track.Absorber(self.abs_len, 890.4, 1.0, 1.00794, 0.0708, 21.8)
        hf_solenoid = RF_Track.Solenoid(self.sol_len, 40, 0.16)
        V.add(hf_solenoid, 0, 0, cell_center, 'center')
        V.add(absorber, 0, 0, cell_center, 'center')
        C_start = RF_Track.Coil(0.5, self.entr_coil_bz, self.entr_coil_r)
        C_end = RF_Track.Coil(0.5, self.exit_coil_bz, self.exit_coil_r)
        V.add(C_start, 0, 0, cell_center-(self.entr_coil_offset), 'exit')
        V.add(C_end, 0, 0, cell_center+(self.exit_coil_offset), 'entrance')
        V.set_s1(start_cell + 4.0)
        track_setup = get_track_setup()
        beam_after_cell = V.track(beam, track_setup)
        if cut:
            beam_after_cell = cut_distribution(beam_after_cell, self.cell_n)
        return beam_after_cell
            

class RotationRF:
    def __init__(self, freq_rot, grad_rot, drift_len, nrot, phase_rot, cell_len):
        self.freq_rot = float(freq_rot*1e6) # Hz
        self.gradient = np.array([grad_rot*1e6]) # V/m
        self.drift_len = drift_len
        self.nrot = nrot
        self.phase = phase_rot
        self.cell_len = float(cell_len)

    def rotate(self, beam, start_volume, bz):
        start_rf_rot = start_volume + float(self.drift_len)
        on_crest = -1
        n_cells = 1
        V = RF_Track.Volume()
        V.set_s0(start_volume)
        V.set_static_Bfield(0, 0.0, bz)
        for i in range(self.nrot):
            SW = RF_Track.SW_Structure(self.gradient, self.freq_rot, self.cell_len, on_crest*n_cells)
            SW.set_phid(float(self.phase))
            SW.set_t0(0.0) # mm/c
            V.add(SW, 0, 0, start_rf_rot, 'entrance')
            start_rf_rot = start_rf_rot + SW.get_length() 
        track_setup = get_track_setup()
        beam_after_rotation = V.track(beam, track_setup)
        return beam_after_rotation


class AccelRF():
    def __init__(self, freq_accel, grad_accel, naccel, cell_len):
        self.freq_accel = float(freq_accel*1e6) # Hz
        self.grad_accel = np.array([grad_accel*1e6]) # V/m
        self.naccel = naccel
        self.cell_len = float(cell_len)

    def accelerate(self, beam, start_volume, bz):
        on_crest = -1
        n_cells = 1
        V = RF_Track.Volume()
        V.set_s0(start_volume)
        V.set_static_Bfield(0, 0.0, bz)
        start_rf = start_volume
        for i in range(self.naccel):
            SW = RF_Track.SW_Structure(self.grad_accel, self.freq_accel, self.cell_len, on_crest*n_cells)
            V.add(SW, 0, 0, start_rf, 'entrance')
            start_rf = start_rf + SW.get_length()
        track_setup = get_track_setup()
        beam_after_acceleration = V.track(beam, track_setup)
        return beam_after_acceleration


if __name__ == "__main__":
    cells = read_json("./FCchannel_025m_RFcav")

    cooling_keys = ['cell_n', 'abs_len', 'entr_coil_bz', 'entr_coil_r', 'entr_coil_offset', 
                        'exit_coil_bz', 'exit_coil_r', 'exit_coil_offset', 'sol_len', 'low_bz']
    rotating_rf_keys = ["freq_rot", "grad_rot", "drift_len", "nrot", "cell_len", "phase_rot"]
    accelerating_rf_keys = ["freq_accel", "grad_accel", "naccel", "cell_len"]

    # bunch6d_init = setup_beam(300, 1.5, 50.0, 145, betax = 0.2839, betay = 0.2782, alphax = -0.04122, alphay = 0.0186, npart=1000, decays_on=True)
    bunch6d_init = setup_beam(300, 1.5, 50.0, 145, betax = 0.28, betay = 0.28, alphax = 0.0, alphay = 0.0, npart=1000, decays_on=True)
    beam_to_track = bunch6d_init
    
    emittz_allcells = []
    emitt4d_allcells = []
    transmission_allcells = []
    espreadRF_allcells = []
    bunch_lengthRF_allcells = []
    pzAccel_allcells = []
    ekinAccel_allcells = []
    
    for json_data in cells:
        RF_Track.rng_set_seed(1245)
        cooling_cell_data = {key: json_data[key] for key in cooling_keys}
        rotating_rf_data = {key: json_data[key] for key in rotating_rf_keys}
        accelerating_rf_data = {key: json_data[key] for key in accelerating_rf_keys}

        cooling_cell = CoolingCell(**cooling_cell_data)
        cooled_beam = cooling_cell.cool_in_cell(beam_to_track, beam_to_track.S, cut=True)
        # utils.get_beam_info_6d(cooled_beam)
        print("Running for cell ", cooling_cell.cell_n)
        if cooling_cell.cell_n == 9:
            final_beam = cooled_beam
        if cooling_cell.cell_n in range(1, 6):
            rotating_rf = RotationRF(**rotating_rf_data)
            beam_rotated = rotating_rf.rotate(cooled_beam, cooled_beam.S, cooling_cell.low_bz)
            # utils.get_beam_info_6d(beam_rotated)
            accelerating_rf = AccelRF(**accelerating_rf_data)
            beam_accelerated = accelerating_rf.accelerate(beam_rotated, beam_rotated.S, cooling_cell.low_bz)
            final_beam = beam_accelerated
        if cooling_cell.cell_n in range(6, 9):
            accelerating_rf = AccelRF(**accelerating_rf_data)
            beam_accelerated = accelerating_rf.accelerate(cooled_beam, cooled_beam.S, cooling_cell.low_bz)
            rotating_rf = RotationRF(**rotating_rf_data)
            beam_rotated = rotating_rf.rotate(beam_accelerated, beam_accelerated.S, cooling_cell.low_bz)
            final_beam = beam_rotated
        if cooling_cell.cell_n > 3:
            final_beam = utils.sigma_cut(final_beam, 3)
        emittz_allcells.append(final_beam.get_info().emitt_z * 1e-3)
        emitt4d_allcells.append(final_beam.get_info().emitt_4d)

        transmission_allcells.append(len(final_beam.get_phase_space())/ len(bunch6d_init.get_phase_space()))
        espreadRF_allcells.append(final_beam.get_info().sigma_E)
        bunch_lengthRF_allcells.append(final_beam.get_info().sigma_t)
        pzAccel_allcells.append(final_beam.get_info().mean_P)
        ekinAccel_allcells.append(final_beam.get_info().mean_K)

        beam_to_track = final_beam
       
    print(emittz_allcells)
    print(emitt4d_allcells)
    print(transmission_allcells)
    print(espreadRF_allcells)
    print(bunch_lengthRF_allcells)
    print(pzAccel_allcells)
    print(ekinAccel_allcells)
