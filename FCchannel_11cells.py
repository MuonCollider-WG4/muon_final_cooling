
import json
import sys
sys.path.append("/Users/elenafol/autophase-test/rf-track-2.1")
import RF_Track
import cooling_utils as utils
import numpy as np
import matplotlib.pyplot as plt
import solenoid
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import pandas as pd

from tabulate import tabulate


def read_json(filename):
    with open(filename, 'r') as file:
        json_data = json.load(file)
        cells =  json_data['cell']
    return cells


def get_track_setup():
    track_setup = RF_Track.TrackingOptions()
    track_setup.dt_mm = 2.0 # mm/c
    track_setup.odeint_algorithm = 'analytic' # analytic' # 'rk2', 'rkf45', 'leapfrog', ...
    track_setup.tt_dt_mm = 5.0  # mm/c, track the emittance every tt_dt_mm steps
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
    # Twiss.sigma_pt = 20
    pz_init = pzinit_mevc

    beam_to_track = RF_Track.Bunch6d(RF_Track.muonmass, Q_nC * RF_Track.nC, +1, pz_init, Twiss, npart)
    if decays_on:
        beam_to_track.set_lifetime(RF_Track.muonlifetime)
    return beam_to_track


def plot_lost_particles(all, lost, title):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # x-px
    axs[0].scatter(all['x'], all['px'], label='Initial')
    axs[0].scatter(lost['x'], lost['px'], color='red', label='lost', marker='x')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('px')
    axs[0].legend()

    # y-py
    axs[1].scatter(all['y'], all['py'], label='Initial')
    axs[1].scatter(lost['y'], lost['py'], color='red', label='lost', marker='x')
    axs[1].set_xlabel('y')
    axs[1].set_ylabel('py')
    axs[1].legend()

    # t-P
    axs[2].scatter(all['t'], all['P'], label='Initial')
    axs[2].scatter(lost['t'], lost['P'], color='red', label='lost', marker='x')
    axs[2].set_xlabel('t')
    axs[2].set_ylabel('P')
    axs[2].legend()

    plt.title(title)
    plt.tight_layout()
    plt.show()


class CoolingCell:
    def __init__(self, cell_n, abs_len, sol_len, low_bz_cool):
        self.cell_n = cell_n
        self.abs_len = abs_len
        self.sol_len = sol_len
        self.low_bz = low_bz_cool
    
    def cool_in_cell(self, beam, start_cell, cut=True):
        V = RF_Track.Volume()
        V.set_s0(start_cell)
        V.set_static_Bfield(0, 0.0, self.low_bz)
        cell_center = start_cell + 2.0
        print("Cell center: ", cell_center)
        absorber = RF_Track.Absorber(self.abs_len, 890.4, 1.0, 1.00794, 0.0708, 21.8)
        hf_solenoid = RF_Track.Solenoid(self.sol_len, 40, 0.16)
        V.add(hf_solenoid, 0, 0, cell_center, 'center')
        V.add(absorber, 0, 0, cell_center, 'center')
        V.set_s1(start_cell + 4.0)
        track_setup = get_track_setup()
        cell_n = self.cell_n
        v_type = 'after_cooling'
        ps_before = pd.DataFrame(beam.get_phase_space('%x %Px %y %Py %t %Pz %id'), columns=['x', 'px', 'y', 'py', 't', 'P', 'id'])
        beam_after_cell = V.track(beam, track_setup)
        ps_after = pd.DataFrame(beam_after_cell.get_phase_space('%x %Px %y %Py %t %Pz %id'), columns=['x', 'px', 'y', 'py', 't', 'P', 'id'])
        lost_df = ps_before[~ps_before['id'].isin(ps_after['id'])]
        # plot_lost_particles(ps_before, lost_df, "Absorber, cell {}".format(self.cell_n))
        print("LOST {} particles: ".format(len(beam.get_phase_space())-len(beam_after_cell.get_phase_space())), len(lost_df))
        
        return beam_after_cell, V

    def cool_in_static_field(self, beam, start_cell, cut=True):
        V = RF_Track.Volume()
        V.set_s0(start_cell)
        V.set_static_Bfield(0, 0.0, 40)
        absorber = RF_Track.Absorber(self.abs_len, 890.4, 1.0, 1.00794, 0.0708, 21.8)
        V.add(absorber, 0, 0, start_cell, 'entrance')
        V.set_s1(start_cell + self.abs_len)
        track_setup = get_track_setup()
        beam_after_cell = V.track(beam, track_setup)
        # utils.plot_tracking_results(beam, beam_after_cell, V)
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

        ps_before = pd.DataFrame(beam.get_phase_space('%x %Px %y %Py %t %Pz %id'), columns=['x', 'px', 'y', 'py', 't', 'P', 'id'])
        beam_after_acceleration = V.track(beam, track_setup)
        ps_after = pd.DataFrame(beam_after_acceleration.get_phase_space('%x %Px %y %Py %t %Pz %id'), columns=['x', 'px', 'y', 'py', 't', 'P', 'id'])
        lost_df = ps_before[~ps_before['id'].isin(ps_after['id'])]
        # plot_lost_particles(ps_before, lost_df, "Accelerating RF")
        print("LOST {} particles: ".format(len(beam.get_phase_space())-len(beam_after_acceleration.get_phase_space())), len(lost_df))

        # solenoid.plot_field(V)
        # utils.plot_tracking_results(beam, beam_after_acceleration, V)
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

        
        ps_before = pd.DataFrame(beam.get_phase_space('%x %Px %y %Py %t %Pz %id'), columns=['x', 'px', 'y', 'py', 't', 'P', 'id'])
        beam_after_rotation = V.track(beam, track_setup)
        ps_after = pd.DataFrame(beam_after_rotation.get_phase_space('%x %Px %y %Py %t %Pz %id'), columns=['x', 'px', 'y', 'py', 't', 'P', 'id'])
        lost_df = ps_before[~ps_before['id'].isin(ps_after['id'])]
        # plot_lost_particles(ps_before, lost_df, "Rotating RF")
        print("LOST {} particles: ".format(len(beam.get_phase_space())-len(beam_after_rotation.get_phase_space())), len(lost_df))
        # solenoid.plot_field(V)
        # utils.plot_tracking_results(beam, beam_after_rotation, V)
        return beam_after_rotation


def plot_phase_space(cooled_beam, title, cell_n):
    phase_space_6d = cooled_beam.get_phase_space('%x %Px %y %Py %t %Pz')
    plt.scatter(phase_space_6d[:, 0], phase_space_6d[:, 1])
    plt.xlabel("x")
    plt.ylabel('Px')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./cell_{0}/x_px_{1}.pdf'.format(cell_n, title))

    plt.scatter(phase_space_6d[:, 2], phase_space_6d[:, 3])
    plt.xlabel("y")
    plt.ylabel('Py')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./cell_{0}/y_py_{1}.pdf'.format(cell_n, title))

    plt.scatter(phase_space_6d[:, 4], phase_space_6d[:, 5])
    plt.xlabel("t")
    plt.ylabel('Pz')
    plt.title(title)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./cell_{0}/t_pz_{1}.pdf'.format(cell_n, title))



class G4bl():
    def t_mmc_to_ns(self, t_mmc):
        t_ns = t_mmc / RF_Track.ns # ns
        return t_ns


    def convert_input(self, beam_path, g4bl_beam_path):
        '''
        Args: 
        beam_path: path rf-track beam to be converted
        g4bl_beam_path: path to save g4bl input beam
        '''
        beam_to_track = RF_Track.Bunch6d()
        beam_to_track.load (beam_path)
        rftrack_particles = beam_to_track.get_phase_space('%x %y %z %Px %Py %Pz %t')
        N = len(rftrack_particles)
        # mm/c to ns
        rftrack_particles[:, -1] = [self.t_mmc_to_ns(t) for t in rftrack_particles[:,-1]]

        with open(g4bl_beam_path, 'w') as file:
            # Write headers
            file.write("#x y z Px Py Pz t PDGid EvNum TrkId Parent weight\n")
            # Generate and write N rows of data
            for i in range(1, N+1):
                # Get the data for the first 6 columns from data_array
                particle_data = rftrack_particles[i-1]
                # Generate the complete row of data
                columns = np.append(particle_data, [int(-13), int(i), int(0), int(0), 1.0])
                # Convert elements to strings and join them with spaces
                data_str = ' '.join(map(str, columns)) + '\n'
                # Write the row to the file
                file.write(data_str)


def get_cell_data(cell_n):
    cells = read_json("./FCchannel_reopt_11cells.json")
    json_data = cells[cell_n]

    cooling_keys = ['cell_n', 'abs_len','sol_len', 'low_bz_cool']
    rotating_rf_keys = ['freq_accel', 'grad_accel', 'drift_len', "nrot",  "cell_len", "phase_rot", "low_bz_rf"]
    accelerating_rf_keys = ["naccel", 'freq_accel', 'grad_accel',"cell_len", "low_bz_rf"]
    cooling_cell_data = {key: json_data[key] for key in cooling_keys}
    rotating_rf_data = {key: json_data[key] for key in rotating_rf_keys}
    accelerating_rf_data = {key: json_data[key] for key in accelerating_rf_keys}
    return cooling_cell_data, rotating_rf_data, accelerating_rf_data


def json_to_latex_table(json_file):
    # Read JSON file
    cells = read_json(json_file)
    
    # Table headers
    headers = ["cell_n", "abs_len", "drift_len", "nrot", "naccel", "freq_accel", "grad_accel", "phase_rot"]

    # Begin LaTeX table
    latex_table = "\\begin{table}[ht]\n"
    latex_table += "\\centering\n"
    latex_table += "\\begin{tabular}{|" + "|".join(["c"] * len(headers)) + "|}\n"
    latex_table += "\\hline\n"

    # Table header
    latex_table += " & ".join(headers) + " \\\\\n"
    latex_table += "\\hline\n"

    # Table data
    for entry in cells:
        row = [str(entry[key]) for key in headers]
        latex_table += " & ".join(row) + " \\\\\n"
    
    # End LaTeX table
    latex_table += "\\hline\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\caption{Caption here}\n"
    latex_table += "\\label{tab:table_label}\n"
    latex_table += "\\end{table}\n"

    return latex_table


if __name__ == "__main__":
    # Example usage
    json_file = '/Users/elenafol/cernbox/FC_rf_studies/rf_track_python_opt/14cells_reopt/FCchannel_reopt_11cells.json'
    # latex_table = json_to_latex_table(json_file)
    # print(latex_table)

    longit_ps_cleaning_rates = [0.01, 0.01, 0.01, 0.02, 0.05, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
    transv_ps_cleaning_rates = [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04]

    RF_Track.rng_set_seed(1869)
    # beam_to_track = RF_Track.Bunch6d()
    # beam_to_track.load("./repopt_sol/cooled_cell_1")
    # utils.get_beam_info_6d(beam_to_track)
    npart_init = 1000

    beam_to_track_init = setup_beam(300, 1.5, 50.0, 145, betax = 0.106, betay = 0.106, alphax = 0.0, alphay = 0.0, npart=npart_init, decays_on=True)
    beam_to_track = beam_to_track_init
    print("Initial Beam: ")
    utils.get_beam_info_6d(beam_to_track)

    emittz_allcells = []
    emitt4d_allcells = []
    emittx_allcells = []
    emitty_allcells = []
    emitt6d_allCells = []
    transmission_allcells = []
    espreadRF_allcells = []
    bunch_lengthRF_allcells = []
    pzAccel_allcells = []
    ekinAccel_allcells = []
    pzCooled_allCells = []
    espreadCooled_allcells = []
    bunch_lengthCooled_allcells = []


    transmission_after_first_rf = []
    transmission_after_secRF = []
    transmission_after_LH = []
    transmission_after_cleaning = []

    bz_peak = []
    bz_const = []
    beam_size_max = []
    beta_max = []
    beam_size_min = []
    beta_min = []
    ekin_before_LH = []
    ekin_after_LH = []
    full_cell_length =[]
    beamsize_4sigmaDiameter = []
    lh_len = []

    
    for cell_n in range(1, 12):
        emitt_start_cell = beam_to_track.get_info().emitt_4d
        betat_m = (2*(beam_to_track.get_info().mean_K*1e-3))/(0.3*4.629)
        RF_Track.rng_set_seed(9876)
        cooling_cell_data, rotating_rf_data, accelerating_rf_data = get_cell_data(cell_n-1)
        
        print(rotating_rf_data)
        print(accelerating_rf_data)
        print("==========Running for cell {} ============".format(cell_n))
        print(cooling_cell_data)
        print("Initial beam: ")
        utils.get_beam_info_6d(beam_to_track)
        cooling_cell = CoolingCell(**cooling_cell_data)
        ldrift, n_cav_rot, n_cav_accel, rot_phase, freq, gradient = \
            rotating_rf_data['drift_len'], rotating_rf_data['nrot'], accelerating_rf_data['naccel'], rotating_rf_data['phase_rot'], \
            accelerating_rf_data['freq_accel'], accelerating_rf_data['grad_accel']
        rotating_rf = RotationRF(freq, gradient, ldrift, n_cav_rot, 0.25)
        accelerating_rf = AccelRF(n_cav_accel, freq, gradient, 0.25)
        if cell_n in range(2,6):
            beam_rotated = rotating_rf.rotate(beam_to_track, beam_to_track.S, rotating_rf_data['low_bz_rf'], rot_phase)
            transmission_after_first_rf.append(len(beam_rotated.get_phase_space())/ 100)
            # plot_phase_space(beam_rotated, title="Cell {0}, After drift and 1st RF, transmission = {1}".format(cell_n, transmission_after_first_rf[-1]))
            beam_accelerated = accelerating_rf.accelerate(beam_rotated, beam_rotated.S, accelerating_rf_data['low_bz_rf'])
            transmission_after_secRF.append(len(beam_rotated.get_phase_space())/ 100)
            # plot_phase_space(beam_accelerated, title="Cell {0}, After 2nd RF, transmission = {1}".format(cell_n, transmission_after_secRF[-1]))
            beam_to_track = beam_accelerated
        if cell_n > 5 :
            print("accelerate, rotate")
            beam_accelerated = accelerating_rf.accelerate(beam_to_track, beam_to_track.S, accelerating_rf_data['low_bz_rf'])
            transmission_after_first_rf.append(len(beam_rotated.get_phase_space())/ 100)
            # plot_phase_space(beam_accelerated, title="Cell {0}, After 1st RF, transmission = {1}".format(cell_n, transmission_after_first_rf[-1]))
            beam_rotated = rotating_rf.rotate(beam_accelerated, beam_accelerated.S, rotating_rf_data['low_bz_rf'], rot_phase)
            transmission_after_secRF.append(len(beam_rotated.get_phase_space())/ 100)
            # plot_phase_space(beam_rotated, title="Cell {0}, After drift and 2nd RF, transmission = {0}".format(cell_n, transmission_after_secRF[-1]))
            beam_to_track = beam_rotated

        pzAccel_allcells.append(beam_to_track.get_info().mean_P)
        espreadRF_allcells.append(beam_to_track.get_info().sigma_E)
        bunch_lengthRF_allcells.append(beam_to_track.get_info().sigma_t)
        ekinAccel_allcells.append(beam_to_track.get_info().mean_K)
        print("Beam before cooling: ")
        utils.get_beam_info_6d(beam_to_track)
        ekin_before_LH.append(beam_to_track.get_info().mean_K)
        cooled_beam, V = cooling_cell.cool_in_cell(beam_to_track, beam_to_track.S, cut=True)
        lh_len.append(cooling_cell.abs_len)
        tt_cooling = V.get_transport_table('%rmax')
        beam_size_max.append(np.max(tt_cooling[:, 0]))
        beam_size_min.append(np.min(tt_cooling[:, 0]))
        beamsize_4sigmaDiameter.append(2*4*(np.sqrt(emitt_start_cell*1e-3 *  betat_m*1e3 * 105.6/cooled_beam.get_info().mean_P))) 
        Za = np.linspace(V.get_s0()[0][2], V.get_s1()[0][2], 1000) # m
        Bz = []
        for Z in Za:
            # get_field() return a list of 2 arrays, each array contains 3 elements
            e_b_fields = V.get_field(0, 0, Z, 0) # x,y,z,t (mm, mm/c)
            Bz.append(e_b_fields[1][2])
        bz_const.append(np.min(Bz))
        bz_peak.append(np.max(Bz))

        transmission_after_LH.append(len(cooled_beam.get_phase_space())/ 100)
        # plot_phase_space(cooled_beam, title="Cell {0}, After Absorber, transmission={1}".format(cell_n, transmission_after_LH[-1]))
        cooled_beam = utils.clean_LongPhasespace(cooled_beam, longit_ps_cleaning_rates[cell_n-1])
        cooled_beam = utils.clean_4dphasespace(cooled_beam, transv_ps_cleaning_rates[cell_n-1])
        ekin_after_LH.append(cooled_beam.get_info().mean_K)
        if cell_n != 1:
            full_cell_length.append(4.0 + rotating_rf.drift_len + 0.25*rotating_rf.nrot + 0.25*accelerating_rf.nrot)
        transmission_after_cleaning.append(len(cooled_beam.get_phase_space())/ 100)
        # plot_phase_space(cooled_beam, title="Cell {0}, After Absorber, cleaned, transmission={1}".format(cell_n, transmission_after_cleaning[-1]))
        beam_to_track = cooled_beam
        print("Transmission: ", len(cooled_beam.get_phase_space())/1000)
        print("Beam Cooled: ")
        utils.get_beam_info_6d(cooled_beam)
        final_beam = cooled_beam

        pzCooled_allCells.append(final_beam.get_info().mean_P)
        emittz_allcells.append(final_beam.get_info().emitt_z * 1e-3)
        emitt4d_allcells.append(final_beam.get_info().emitt_4d)
        espreadCooled_allcells.append(final_beam.get_info().sigma_E)
        bunch_lengthCooled_allcells.append(final_beam.get_info().sigma_t)

        emittx_allcells.append(final_beam.get_info().emitt_x)
        emitty_allcells.append(final_beam.get_info().emitt_y)
        emitt6d_allCells.append(final_beam.get_info().emitt_6d)

        transmission_allcells.append(len(final_beam.get_phase_space())/npart_init)
        
        
    print("Bz peak [T]: ", bz_peak)
    print("Bz const [T]: ", bz_const)
    print("Beam rms max [mm]: ", beam_size_max)
    print("Beam rms min [mm]:", beam_size_min)
    print("Ekin before cooling [MeV]: ", ekin_before_LH)
    print("Ekin after cooling [MeV]: ", ekin_after_LH)
    print("Full length of each stage: ", full_cell_length)

    # print(transmission_after_first_rf)
    # print(transmission_after_secRF)
    # print(transmission_after_LH)
    # print(transmission_after_cleaning)
    
    print("Transmission: ", transmission_allcells)
    print("6D Emittance: ", emitt6d_allCells)
    print("Longit. Emittance: ", emittz_allcells)
    print("4D Emittance: ", emitt4d_allcells)
    # print("Emittances X", emittx_allcells)
    # print("Emittances Y", emitty_allcells)
    print("sigma E", espreadRF_allcells)
    print("sigma t", bunch_lengthRF_allcells)
    print("Pz, accelerated: ", pzAccel_allcells)
    print("Ekin, accelerated: ", ekinAccel_allcells)

    table_beam_params = [pzAccel_allcells, espreadRF_allcells, bunch_lengthRF_allcells, pzCooled_allCells, espreadCooled_allcells, bunch_lengthCooled_allcells, emittz_allcells, emitt4d_allcells, emitt6d_allCells, transmission_after_cleaning]
    print(tabulate(np.array(table_beam_params).T, tablefmt="latex", floatfmt=".1f"))

    # latex_table = json_to_latex_table(json_file)
    # print(latex_table)

    # table_aperture = [range(1, 13), beamsize_4sigmaDiameter, lh_len, ekin_before_LH, ekin_after_LH]
    # print(tabulate(np.array(table_aperture).T, tablefmt="latex", floatfmt=".2f"))


