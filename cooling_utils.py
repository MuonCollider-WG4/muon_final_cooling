import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, constants
from sklearn.covariance import EllipticEnvelope, MinCovDet
muonmass = constants.physical_constants['muon mass energy equivalent in MeV'][0]

def clean_6dphasespace(beam_to_clean, contamination_rate):
    algorithm =  EllipticEnvelope(contamination=contamination_rate)
    X = beam_to_clean.get_phase_space('%x %Px %y %Py %t %Pz')
    algorithm.fit(X)
    y_pred = algorithm.fit(X).predict(X)
    bunch_filtered = beam_to_clean.get_phase_space()[np.where(y_pred != -1)[0]]
    beam_to_clean.set_phase_space(bunch_filtered)
    return beam_to_clean


def clean_robust_cov(beam):
    phase_space_emitt = beam.get_phase_space('%x %Px %y %Py %t %Pz') # %t %E') 
    # estimate robust covariance
    cov6d = MinCovDet(random_state=0, assume_centered=False).fit(phase_space_emitt)
    cleaned_phase_space_6d = beam.get_phase_space()[np.where(cov6d.support_ == True)]
    beam.set_phase_space(cleaned_phase_space_6d)
    return  beam


def clean_4dphasespace(beam_to_clean, contamination_rate):
    algorithm =  EllipticEnvelope(contamination=contamination_rate)
    ps6d = beam_to_clean.get_phase_space()
    X = beam_to_clean.get_phase_space('%x %Px %y %Py')
    algorithm.fit(X)
    y_pred = algorithm.fit(X).predict(X)
    bunch_filtered = ps6d[np.where(y_pred != -1)[0]]
    beam_to_clean.set_phase_space(bunch_filtered)
    return beam_to_clean


def clean_LongPhasespace(beam_to_clean, contamination_rate):
    algorithm =  EllipticEnvelope(contamination=contamination_rate)
    X = beam_to_clean.get_phase_space('%t %Pz')
    algorithm.fit(X)
    y_pred = algorithm.fit(X).predict(X)
    bunch_filtered = beam_to_clean.get_phase_space()[np.where(y_pred != -1)[0]]
    beam_to_clean.set_phase_space(bunch_filtered)
    return beam_to_clean


def sigma_cut(final_beam, sigma_cut, plot=False):
    final_beam_phase_space = final_beam.get_phase_space('%x %Px %y %Py %t %Pz')
    final_phase_space_df = pd.DataFrame(final_beam_phase_space,
                    columns=['X', 'Px', 'Y', 'Py', 't', 'Pt'])
    cut_final_idx = final_phase_space_df.index[
        (np.abs(stats.zscore(final_phase_space_df)) < sigma_cut).all(axis=1)]
    cut_final_phasespace = final_phase_space_df.iloc[cut_final_idx]
    if plot: 
        print(cut_final_phasespace.values.shape)
        print(type(cut_final_phasespace.values))
        plt.xlabel('x')
        plt.ylabel('xp')
        plt.scatter(final_beam_phase_space[:,0], final_beam_phase_space[:,1], label='initial beam')
        plt.scatter(cut_final_phasespace.values[:,0], cut_final_phasespace.values[:,1], label='filtered beam')
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.xlabel('y')
        plt.ylabel('py')
        plt.scatter(final_beam_phase_space[:,2], final_beam_phase_space[:,3], label='initial beam')
        plt.scatter(cut_final_phasespace.values[:,2], cut_final_phasespace.values[:,3], label='filtered beam')
        plt.legend()
        plt.tight_layout()
        plt.show()
    final_beam.set_phase_space(final_beam.get_phase_space()[cut_final_idx])
    return final_beam


def get_beam_info_6d(final_beam_6d):
    final_beam_info = final_beam_6d.get_info()
    final_pz = final_beam_info.mean_P
    final_emitt_4d = final_beam_info.emitt_4d
    final_emitt_z = final_beam_info.emitt_z/1e3
    bunch_len = final_beam_info.sigma_t

    print( "Mean Ekin = ", round(final_beam_info.mean_K, 2),
        "Pz = ", round(final_pz,2), "SigmaT = ", round(bunch_len,2), "Emitt_4d = ", round(final_emitt_4d,2),  "Emitt_z = ", round(final_emitt_z, 2),
        "Sigma Ekin [MeV]= ", round(final_beam_info.sigma_E, 2))

def compute_emittance_from_phasespace(beam):
    phase_space = beam.get_phase_space('%x %Px %y %Py %t') # 6d
    # phase_space = beam.get_phase_space('%X %Px %Y %Py %Z') # 6dt
    t_sorted = phase_space[phase_space[:, 4].argsort()]
    det = np.linalg.det(np.cov(t_sorted[:, :4], rowvar=False))
    emitt4d_full_bunch =  pow(det, 0.25) / muonmass * 1e3
    return  emitt4d_full_bunch


def compute_emittz_from_phase_space(phase_space):
    p = phase_space[:, 5]
    t = phase_space[:, 4]
    Pz = p / np.hypot(phase_space[:, 1], phase_space[:, 3])
    Px = phase_space[:, 1] * Pz 
    Py = phase_space[:, 3] * Pz
    E = np.sqrt(muonmass**2 + Px**2 + Py**2 + Pz**2)
    emitt_z = np.sqrt(np.linalg.det(np.cov(np.array([t, E])))) / muonmass
    return emitt_z


def plot_tracking_results(beam_to_track, final_beam, V):
    plt.rcParams.update({'font.size': 14})
    transport_table = V.get_transport_table('%mean_Z %mean_Pz %sigma_Pz %emitt_4d %emitt_z %beta_x %beta_y %alpha_x %alpha_y %emitt_x %emitt_y %N %sigma_Z')
    init_beam_phase_space = beam_to_track.get_phase_space('%x %xp %y %py %t %Pz')
    final_beam_phase_space = final_beam.get_phase_space('%x %xp %y %py %t %Pz')

    # t is arrival time at S
    plt.xlabel('t [m/c]')
    plt.ylabel('Pz [MeV/c]')
    plt.scatter(init_beam_phase_space[:,4]/1e3, init_beam_phase_space[:,5], label='initial beam')
    plt.scatter(final_beam_phase_space[:,4]/1e3, final_beam_phase_space[:,5], label='final beam')
    plt.legend()
    plt.tight_layout()
    plt.show()

    
    fig, ax = plt.subplots()
    ax.plot(transport_table[:,0]/1e3, transport_table[:,1], label='Pz')
    ax.fill_between(transport_table[:,0]/1e3, transport_table[:,1]-transport_table[:,2], transport_table[:,1]+transport_table[:,2], alpha=0.2, label="Sigma Pz")
    ax.set_xlabel('mean Z [m]')
    ax.set_ylabel('Pz [MeV/c]')
    plt.legend()
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots()
    ax.plot(transport_table[:,0]/1e3, transport_table[:,-1], label='Bunch length')
    ax.set_xlabel('mean Z [m]')
    ax.set_ylabel('sigmat [mm]')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.xlabel('mean Z [m]')
    plt.ylabel('emitt 4d [microns]')
    
    plt.plot(transport_table[:,0]/1e3,  transport_table[:,9], linestyle = '--',label='emitt x')
    plt.plot(transport_table[:,0]/1e3, transport_table[:,10], linestyle = '--',label='emitt y')
    plt.plot(transport_table[:,0]/1e3, transport_table[:,3], label='emitt 4d')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.xlabel('S [m]')
    plt.ylabel('emitt z [mm]')
    plt.plot(transport_table[:,0]/1e3, transport_table[:,4]/1e3, label='emitt z')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.xlabel('mean Z [m]')
    plt.ylabel('Twiss')
    plt.plot(transport_table[:,0]/1e3, transport_table[:,5], label='beta x')
    plt.plot(transport_table[:,0]/1e3, transport_table[:,7], label='alpha x')
    plt.legend()
    plt.tight_layout()
    plt.show()


def linearize_phase_space(phase_space, plot=False):
    t = phase_space[:, 4]
    p = phase_space[:, 5]
    # corelation degrees in longitudinal phase space
    fit_coeff = np.polyfit(t, p, 3)
    p_lin = p + np.mean(p) - np.polyval(fit_coeff, t)
    uncorr_sigmae = np.std(p_lin)
    lin_phase_space = phase_space.copy()
    lin_phase_space[:,5] = p_lin
    if plot:
        plt.xlabel('ct [m]')
        plt.ylabel('Pz [MeV/c]')
        plt.scatter(phase_space[:,4]/1e3, phase_space[:,5], label='original')
        plt.scatter(lin_phase_space[:,4]/1e3, lin_phase_space[:,5], label='linearized')
        plt.plot(t/1e3, np.polyval(fit_coeff, t), color='green', label='fit')
        plt.legend()
        plt.show()
    t_p_fit = stats.linregress(t, p)
    return lin_phase_space, uncorr_sigmae, t_p_fit.slope