# Packages
import numpy as np
import pandas as pd
import coolpy as cp
import subprocess
from py2g4bl_match_helpers import P, CellBeamGen, compute_emit_from_phasespace, readfile, E_to_dpp_to_dE, plot_solenoid

def generate_ref_particles(MD_solenoid_list, rf_lists):
    rf_rot_grad_arr, rf_grad_arr, acc_freq_arr, rot_freq_arr, acc_phase_arr, rot_phase_arr = rf_lists

    # Global Parameters as defined by the genetic algorithm
    Energies = [123.4874061 , 123.34209297, 100.41454084,  85.48562669,  74.47974494,  51.82242803,  32.01412767,  18.28142271, 17.9391355 ,  14.70422686]
    AbsorberLengths = np.array([1.27736269, 1.32518469, 1.10521782, 1.13798138, 0.81575282, 0.43656683, 0.30241914, 0.10448507, 0.11920284, 0.09100977])
    EnergySpreads = [5.41686927, 5.81701861, 5.25003438, 5.06819512, 5.65785391, 4.36047814, 2.39677483, 1.29680007, 1.25195259, 1.20749703]

    Length_max = np.sum(AbsorberLengths)

    # Target emittances as achieved by the simple G4BL setup
    EmitT_Target = np.array([0.0002466918788431969,  0.0002026040562576625,  0.00016493170280864206,  0.00012640334072074876,  0.00010316360751668846,  8.722476322836085e-05,  6.031231528154821e-05,  4.5028352240203934e-05,  3.15684664407679e-05,  2.221710939561845e-05])
    EmitL_Target = np.array([0.0018836404178010724,  0.0022763763903832462,  0.0028338086313568615,  0.003975051209763533,  0.005130191267191367,  0.006728425742772072,  0.011819326472499425,  0.020757848567312387,  0.03924855255649506,  0.07171590993046725])

    # Solenoid parameters
    nSheet = 5
    offset = 0.0

    ### Radius
    L_radius_in  = 0.1 #[m] Low Field
    L_radius_out = 0.3 #[m]
    M_radius_in  = 0.1 # Matching field
    M_radius_out = 0.3
    H_radius_in  = 0.03 # High Field
    H_radius_out = 0.09

    Hcorr1_radius_out = 0.135 #outer most down and up stream correctors
    Hcorr2_radius_out = 0.125 #middle  down and up stream correctors
    Hcorr3_radius_out = 0.115 #inner most down and up stream correctors

    ### Length
    L_Lsol = 1 #[m]
    L_Lsol_short = 0.5 # 0.5
    M_Lsol = 0.2
    H_Lsol_list = [abslen + 2*0.12 for abslen in AbsorberLengths]
    Hcorr_Lsol = 0.012

    #### Current density ####
    L_current_density = 17.  #[A/mm^2]
    H_current_density = 535.3 #A mm^-2
    H2_current_density = -17 #A mm^-2
    M_current_density = 50 #A mm^-2

    NU_matchers = 4
    ND_matchers = 10

    L_Lgap = 0.4
    L_Hgap = 0.1
    L_MUgap = 0.1
    L_MDgap = 0.25+0.1
    L1_pos = 0 #[m]

    # Setting up first cell (unique)
    MU_pos_arr = [(L_Lsol/2 + L_Lgap + L_MUgap*i + M_Lsol*i + M_Lsol/2) for i in range(NU_matchers)]
    H1_pos = MU_pos_arr[-1] + M_Lsol/2 + L_Hgap + H_Lsol_list[0]/2 + Hcorr_Lsol*3

    H_pos_arr = np.zeros(10)
    H_pos_arr[0] = H1_pos

    # Finds the high field solenoid and downstream matcher positions iteratively
    MD_pos_arr_all = []
    for ncell in range(1, 10):
        H_pos = H_pos_arr[ncell-1]
        if H_pos != 0:
            MD_pos_arr = [(H_pos + Hcorr_Lsol*3 + H_Lsol_list[ncell-1]/2 + L_Hgap + L_MDgap*i + M_Lsol*i + M_Lsol/2) for i in range(ND_matchers)]
        H_pos_new = MD_pos_arr[-1] + M_Lsol/2 + L_Hgap + H_Lsol_list[ncell]/2 + Hcorr_Lsol*3
        MD_pos_arr_all.append(np.array(MD_pos_arr))
        H_pos_arr[ncell] = H_pos_new
    MD_pos_arr_all = np.array(MD_pos_arr_all)

    # Same but for the high-field correctors
    Hcorr_rad_arr = [Hcorr1_radius_out, Hcorr2_radius_out, Hcorr3_radius_out]

    HcorrU_pos_arr_all = []
    HcorrD_pos_arr_all = []

    for ncell in range(10):
        HcorrU_arr = [(H_pos_arr[ncell] - (H_Lsol_list[ncell]*0.5 + Hcorr_Lsol*(3-i)) + Hcorr_Lsol*0.5) for i in range(3)]
        HcorrD_arr = [(H_pos_arr[ncell] + (H_Lsol_list[ncell]*0.5 + Hcorr_Lsol*(3-i)) - Hcorr_Lsol*0.5) for i in range(3)]
        HcorrU_pos_arr_all.append(np.array(HcorrU_arr))
        HcorrD_pos_arr_all.append(np.array(HcorrD_arr))

    fieldflip = np.tile([1, -1], 5)

    # First Cell
    L1 = cp.SolenoidSheet(current_density=L_current_density, radius_inner=L_radius_in, radius_outer=L_radius_out, rho=offset, L_sol=L_Lsol, nSheet=nSheet, position=L1_pos)
    MU1_list = [cp.SolenoidSheet(current_density=M_current_density, radius_inner=M_radius_in, radius_outer=M_radius_out, rho=0.0, L_sol=M_Lsol, nSheet=nSheet, position=MU_pos_arr[m]) for m in [0, 2, 3]]

    # Matching solenoids
    MD_solenoid_list = []
    for ncell in range(9):
        for m in range(ND_matchers):
            MD_list = cp.SolenoidSheet(current_density=M_current_density, radius_inner=M_radius_in, radius_outer=M_radius_out, rho=0.0, L_sol=M_Lsol, nSheet=nSheet, position=MD_pos_arr_all[ncell][m])
            MD_solenoid_list.append(MD_list)
            
    # High field solenoids and correctors
    H_solenoid_list = [cp.SolenoidSheet(current_density=fieldflip[ncell]*H_current_density, radius_inner=H_radius_in, radius_outer=H_radius_out, rho=offset, L_sol=H_Lsol_list[ncell], nSheet=nSheet, position=H_pos_arr[ncell]) for ncell in range(10)]

    HcorrU_solenoid_list = []
    HcorrD_solenoid_list = []
    for ncell in range(10):
        for corr in range(3):
            HcorrU = cp.SolenoidSheet(current_density=fieldflip[ncell]*H_current_density, radius_inner=H_radius_in, radius_outer=Hcorr_rad_arr[corr], rho=0.0, L_sol=Hcorr_Lsol, nSheet=nSheet, position=HcorrU_pos_arr_all[ncell][corr])
            HcorrD = cp.SolenoidSheet(current_density=fieldflip[ncell]*H_current_density, radius_inner=H_radius_in, radius_outer=Hcorr_rad_arr[corr], rho=0.0, L_sol=Hcorr_Lsol, nSheet=nSheet, position=HcorrD_pos_arr_all[ncell][corr])
            HcorrU_solenoid_list.append(HcorrU)
            HcorrD_solenoid_list.append(HcorrD)

    MU_optimized_params = [34.07907333777548, 3.3219755405093148, 51.652979865357764]
    for m, matcher in enumerate(MU1_list):
        matcher.current_density = MU_optimized_params[m]

    sol_list = [L1] + MU1_list + MD_solenoid_list + H_solenoid_list + HcorrU_solenoid_list + HcorrD_solenoid_list

    setup = f'''
    g4ui when=4 "/vis/viewer/set/background 1 1 1"
    param disable=Decay   #Whether muons decay
    param stochastics=1
    physics QGSP_BIC disable=$disable
    param cell_end={(H_pos_arr[-1]+H_Lsol_list[-1]/2)*1E3}
    param zstep=50
    '''

    absorber_list = []
    highcoil_list = []
    for cellno in [0,1,2,3,4,5,6,7,8,9]:
        absorbers = f'''
    tube abs{cellno} length={AbsorberLengths[cellno]*1E3} material=LH2 innerRadius=0 outerRadius=25 color=1,1,1,1
    place abs{cellno} z={H_pos_arr[cellno]*1E3:.2f}'''
        absorber_list.append(absorbers)

    for cellno in range(10):
        highcoils = f'''
    coil H{cellno+1}_C innerRadius=30 outerRadius=90 length={H_solenoid_list[cellno].L_sol*1E3} nSheets=3 tolerance=1 maxZ=7000'''
        highcoil_list.append(highcoils)

    coils = f'''
    ### Low Field
    coil L1_C innerRadius=100 outerRadius=300 length={L1.L_sol*1E3} nSheets=3 tolerance=1 maxZ=7000

    ### High Field
    coil Hcorr1_C innerRadius=30 outerRadius=135 length=12 nSheets=3 tolerance=1 maxZ=7000
    coil Hcorr2_C innerRadius=30 outerRadius=125 length=12 nSheets=3 tolerance=1 maxZ=7000
    coil Hcorr3_C innerRadius=30 outerRadius=115 length=12 nSheets=3 tolerance=1 maxZ=7000

    # Matchers
    coil M_C innerRadius=100 outerRadius={M_radius_out*1E3} length={M_Lsol*1E3} nSheets=10 tolerance=1 maxZ=7000
    '''

    low_solenoid = f'''
    solenoid L1 coil=L1_C current=17.0 color=0.929,0.616,0.035,1
    place L1 z=0'''

    high_solenoids = []
    for cellno, Hsol in enumerate(H_solenoid_list):
        if Hsol.current_density > 0:
            color='1,0,0,0.5'
        else:
            color='0,0,1,0.5'
        highsolenoid = f'''
    solenoid H{cellno+1} coil=H{cellno+1}_C current={Hsol.current_density} color={color}
    place H{cellno+1} z={Hsol.position*1E3}'''
        high_solenoids.append(highsolenoid)
        
    HCorrU_solenoids = []
    for u, hcorr in enumerate(HcorrU_solenoid_list):
            if hcorr.current_density > 0:
                    color='1,0,0,0.5'
            else:
                    color='0,0,1,0.5'
            HCU = f'''
    solenoid H{u//3+1}corr{u%3+1}_U coil=Hcorr{u%3+1}_C current={hcorr.current_density} color={color}
    place H{u//3+1}corr{u%3+1}_U z={hcorr.position*1E3}'''
            HCorrU_solenoids.append(HCU)
            
    HCorrD_solenoids = []
    for d, hcorr in enumerate(HcorrD_solenoid_list):
            if hcorr.current_density > 0:
                    color='1,0,0,0.5'
            else:
                    color='0,0,1,0.5'
            HCD = f'''
    solenoid H{d//3+1}corr{d%3+1}_U coil=Hcorr{d%3+1}_C current={hcorr.current_density} color={color}
    place H{d//3+1}corr{d%3+1}_U z={hcorr.position*1E3}'''
            HCorrD_solenoids.append(HCD)


    MU_pos_arr = [1.0, 1.6, 1.9000000000000004]
    MU_solenoids = []
    for i in range(3):
        MU = f'''
    solenoid MU{i+1} coil=M_C current={MU1_list[i].current_density} color=1,0,1,1
    place MU{i+1} z={MU_pos_arr[i]*1E3}'''
        MU_solenoids.append(MU)

    MD_solenoids = []
    for md, Matcher in enumerate(MD_solenoid_list):
        MD = f'''
    solenoid MD{md//10+1}_{md%10+1} coil=M_C current={Matcher.current_density} color=1,0,1,1
    place MD{md//10+1}_{md%10+1} z={Matcher.position*1E3}'''
        MD_solenoids.append(MD)


    # With timeOffset=0

    acc_rot_swap_cell = 6

    rf_list = []
    for rf, Matcher in enumerate(MD_solenoid_list):
        rf_no, cell_no = rf%10, rf//10

        if cell_no < acc_rot_swap_cell: #Rotation first, then acceleration
            if rf_no >= 9: #There is no 10th solenoid
                pass
            else:
                rf1_pos= Matcher.position+0.25+0.05/2
                if rf_no < 4: #Rotational cavities
                    RF = f'''
        pillbox rf{cell_no+1}_{rf_no+1} innerLength=250  maxGradient={rf_rot_grad_arr[cell_no]} frequency={1E-3*rot_freq_arr[cell_no]} innerRadius=160 phaseAcc={rot_phase_arr[cell_no]} win1Thick=0 win2Thick=0 color=1,0.5,0.5,1
        place rf{cell_no+1}_{rf_no+1} z={rf1_pos*1E3}'''
                if rf_no >= 4:
                    RF = f'''
        pillbox rf{cell_no+1}_{rf_no+1} innerLength=250  maxGradient={rf_grad_arr[cell_no]} frequency={1E-3*acc_freq_arr[cell_no]} innerRadius=160 phaseAcc={acc_phase_arr[cell_no]} win1Thick=0 win2Thick=0 color=1,0.5,0.5,1
        place rf{cell_no+1}_{rf_no+1} z={rf1_pos*1E3}'''
                rf_list.append(RF)
        elif cell_no >= acc_rot_swap_cell:
            if rf_no >= 9: #There is no 10th solenoid
                pass
            else:
                rf1_pos= Matcher.position+0.25+0.05/2
                if rf_no >= 4: #Rotational cavities
                    RF = f'''
        pillbox rf{cell_no+1}_{rf_no+1} innerLength=250  maxGradient={rf_rot_grad_arr[cell_no]} frequency={1E-3*rot_freq_arr[cell_no]} innerRadius=160 phaseAcc={rot_phase_arr[cell_no]} win1Thick=0 win2Thick=0 color=1,0.5,0.5,1
        place rf{cell_no+1}_{rf_no+1} z={rf1_pos*1E3}'''
                if rf_no < 4:
                    RF = f'''
        pillbox rf{cell_no+1}_{rf_no+1} innerLength=250  maxGradient={rf_grad_arr[cell_no]} frequency={1E-3*acc_freq_arr[cell_no]} innerRadius=160 phaseAcc={acc_phase_arr[cell_no]} win1Thick=0 win2Thick=0 color=1,0.5,0.5,1
        place rf{cell_no+1}_{rf_no+1} z={rf1_pos*1E3}'''
                rf_list.append(RF)



    track_p0 = f'''
    param ref_p={P((Energies[0])*1E6)*1E-6}

    zntuple output file=2208_Ver0.00_MatchingCell10_p0.txt format=FOR009.DAT zloop=0:$cell_end:$zstep referenceParticle=1
    reference particle=mu+ beamZ=0 referenceMomentum=$ref_p beamX=0.0 beamY=0.0 beamXp=0.0 beamYp=0.0 noEfield=0 noEloss=0
    beam ascii file=050625BeamInput_Reference.txt beamZ=0.0 beamX=0.0 beamY=0.0 beamXp=0.0 beamYp=0.0 lastEvent={3} # user provided file with particle data
    trackcuts keep=mu+
    '''
        

    with open(f'Cell10_Ver0.00_141125_ref.g4bl', "w") as f:
            f.write(setup)
            f.write(coils)
            f.write(low_solenoid)
            for cellno in range(len(absorber_list)):
                    f.write(absorber_list[cellno])
            f.write('''
                    ''')
            for cellno in range(10):
                    f.write(highcoil_list[cellno])
            f.write('''
                    ''')
            for cellno in range(10):
                    f.write(high_solenoids[cellno])
            f.write('''
                    ''')
            for hcorr in HCorrU_solenoids:
                    f.write(hcorr)
            for hcorr in HCorrD_solenoids:
                    f.write(hcorr)
            f.write('''
                    ''')
            for MU in MU_solenoids:
                    f.write(MU)
            f.write('''
                    ''')
            for MD in MD_solenoids:
                    f.write(MD)
            f.write('''
                    ''')
            for rf in rf_list:
                    f.write(rf)
            f.write('''
                    ''')
            #f.write(track_all)
            f.write(track_p0)
            #f.write(track_p2)

    subprocess.run([r"C:\Program Files\Muons, Inc\G4beamline\bin\g4bl.exe", f"Cell10_Ver0.00_141125_ref.g4bl"])