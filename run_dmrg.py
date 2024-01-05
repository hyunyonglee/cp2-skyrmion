import numpy as np
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg, tebd
import argparse
import logging.config
import os
import os.path
import h5py
from tenpy.tools import hdf5_io
import model

def ensure_dir(f):
    d=os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)
    return d

def measurements(psi):
    
    # Measurements
    Sx = psi.expectation_value("Sx")
    Sy = psi.expectation_value("Sy")
    Sz = psi.expectation_value("Sz")
    
    SxSx = psi.expectation_value("Sx Sx")
    SySy = psi.expectation_value("Sy Sy")
    SzSz = psi.expectation_value("Sz Sz")
    
    Q1 = - SxSx + SySy
    Q2 = np.sqrt(1./3.) * ( 2.*SzSz - SxSx - SySy )
    Q3 = psi.expectation_value("Sx Sy") + psi.expectation_value("Sy Sx")
    Q4 = - psi.expectation_value("Sz Sx") + psi.expectation_value("Sx Sz")
    Q5 = psi.expectation_value("Sy Sz") + psi.expectation_value("Sz Sy")

    return np.real(Sx), np.real(Sy), np.real(Sz), np.real(Q1), np.real(Q2), np.real(Q3), np.real(Q4), np.real(Q5)


def write_data( psi, E, Sx, Sy, Sz, Q1, Q2, Q3, Q4, Q5, Lx, Ly, delta, D, h, path ):

    ensure_dir(path+"/observables/")
    ensure_dir(path+"/mps/")

    data = {"psi": psi}
    with h5py.File(path+"/mps/psi_Lx_%d_Ly_%d_delta_%.2f_D_%.2f_h_%.2f.h5" % (Lx,Ly,delta,D,h), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

    file_Sx = open(path+"/observables/Sx.txt","a", 1)
    file_Sy = open(path+"/observables/Sy.txt","a", 1)
    file_Sz = open(path+"/observables/Sz.txt","a", 1)
    file_Q1 = open(path+"/observables/Q1.txt","a", 1)
    file_Q2 = open(path+"/observables/Q2.txt","a", 1)
    file_Q3 = open(path+"/observables/Q3.txt","a", 1)
    file_Q4 = open(path+"/observables/Q4.txt","a", 1)
    file_Q5 = open(path+"/observables/Q5.txt","a", 1)
        
    file_Sx.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Sx)) + " " + "\n")
    file_Sy.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Sy)) + " " + "\n")
    file_Sz.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Sz)) + " " + "\n")
    file_Q1.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Q1)) + " " + "\n")
    file_Q2.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Q2)) + " " + "\n")
    file_Q3.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Q3)) + " " + "\n")
    file_Q4.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Q4)) + " " + "\n")
    file_Q5.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + "  ".join(map(str, Q5)) + " " + "\n")
    
    file_Sx.close()
    file_Sy.close()
    file_Sz.close()
    file_Q1.close()
    file_Q2.close()
    file_Q3.close()
    file_Q4.close()
    file_Q5.close()
    
    #
    file = open(path+"/observables.txt","a", 1)    
    file.write(repr(delta) + " " + repr(h) + " " + repr(D) + " " + repr(E) + " " + repr(np.mean(Sx)) + " " + repr(np.mean(Sy)) + " " + repr(np.mean(Sz)) + " " + repr(np.mean(Q1)) + " " + repr(np.mean(Q2)) + " " + repr(np.mean(Q3)) + " " + repr(np.mean(Q4)) + " " + repr(np.mean(Q5)) + " " + "\n")
    file.close()

    

if __name__ == "__main__":
    
    current_directory = os.getcwd()

    conf = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {'custom': {'format': '%(levelname)-8s: %(message)s'}},
    'handlers': {'to_file': {'class': 'logging.FileHandler',
                             'filename': 'log',
                             'formatter': 'custom',
                             'level': 'INFO',
                             'mode': 'a'},
                'to_stdout': {'class': 'logging.StreamHandler',
                              'formatter': 'custom',
                              'level': 'INFO',
                              'stream': 'ext://sys.stdout'}},
    'root': {'handlers': ['to_stdout', 'to_file'], 'level': 'DEBUG'},
    }
    logging.config.dictConfig(conf)

    # parser for command line arguments
    parser=argparse.ArgumentParser()
    parser.add_argument("--Lx", default='2', help="Length of cylinder")
    parser.add_argument("--Ly", default='2', help="Length of cylinder")
    parser.add_argument("--J1", default='1.0', help="(Ferromagnetic) nn Heisenberg coupling")
    parser.add_argument("--J2", default='0.61803398875', help="(Ferromagnetic) nnn Heisenberg coupling")
    parser.add_argument("--delta", default='1.0', help="Anisotropy")
    parser.add_argument("--D", default='0.0', help="Easy-plane anisotropy")
    parser.add_argument("--h", default='0.0', help="Magnetic field")
    parser.add_argument("--chi", default='20', help="Bond dimension")
    parser.add_argument("--init_state", default='up', help="Initial state")
    parser.add_argument("--RM", default=None, help="path for saving data")
    parser.add_argument("--max_sweep", default='50', help="Maximum number of sweeps")
    parser.add_argument("--path", default=current_directory, help="path for saving data")
    args=parser.parse_args()

    # parameters
    Lx = int(args.Lx)
    Ly = int(args.Ly)
    J1 = float(args.J1)
    J2 = float(args.J2)
    delta = float(args.delta)
    D = float(args.D)
    h = float(args.h)
    chi = int(args.chi)
    init_state = args.init_state
    RM = args.RM
    max_sweep = int(args.max_sweep)
    path = args.path

    # model parameters    
    model_params = {
    "Lx": Lx,
    "Ly": Ly,
    "J1": J1,
    "J2": J2,
    "delta": delta,
    "D": D,
    "h": h
    }

    CP2_SkX = model.CP2_SKYRMION(model_params)

    # initial state
    product_state = [init_state] * (Lx * Ly)
    psi = MPS.from_product_state(CP2_SkX.lat.mps_sites(), product_state, bc=CP2_SkX.lat.bc_MPS)

    if RM == 'random':
        TEBD_params = {'N_steps': 10, 'trunc_params':{'chi_max': 30}, 'verbose': 0}
        eng = tebd.RandomUnitaryEvolution(psi, TEBD_params)
        eng.run()
        psi.canonical_form() 

    dmrg_params = {
    # 'mixer': True,  # setting this to True helps to escape local minima
    'mixer' : dmrg.SubspaceExpansion,
    'mixer_params': {
        'amplitude': 1.e-4,
        'decay': 2.0,
        'disable_after': 20
    },
    'trunc_params': {
        'chi_max': chi,
        'svd_min': 1.e-9
    },
    'chi_list': { 0: 8, 5: 16, 10: 32, 15: 64, 20: chi },
    'max_E_err': 1.0e-9,
    'max_S_err': 1.0e-9,
    'max_sweeps': max_sweep,
    'combine' : True
    }

    # ground state
    eng = dmrg.TwoSiteDMRGEngine(psi, CP2_SkX, dmrg_params)
    E, psi = eng.run()  # equivalent to dmrg.run() up to the return parameters.
    psi.canonical_form() 

    Sx, Sy, Sz, Q1, Q2, Q3, Q4, Q5 = measurements(psi)
    write_data( psi, E, Sx, Sy, Sz, Q1, Q2, Q3, Q4, Q5, Lx, Ly, delta, D, h, path )