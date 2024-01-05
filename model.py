# Copyright 2024 Hyun-Yong Lee

from tenpy.models.lattice import Triangular
from tenpy.models.model import CouplingModel, MPOModel
from tenpy.tools.params import Config
from tenpy.networks.site import SpinSite
import numpy as np
__all__ = ['CP2_SKYRMION']


class CP2_SKYRMION(CouplingModel,MPOModel):
    
    def __init__(self, model_params):
        
        # 0) read out/set default parameters 
        if not isinstance(model_params, Config):
            model_params = Config(model_params, "CP2_SKYRMION")
        Lx = model_params.get('Lx', 1)
        Ly = model_params.get('Ly', 2)
        J1 = model_params.get('J1', 1.)
        J2 = model_params.get('J2', 0.)
        delta = model_params.get('delta', 1.)
        D = model_params.get('D', 0.)
        h = model_params.get('h', 0.)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', 'periodic')
        
        site = SpinSite( S=1, conserve=None, sort_charge=None )
        site.multiply_operators(['Sx','Sx'])
        site.multiply_operators(['Sy','Sy'])
        site.multiply_operators(['Sz','Sz'])
        
        site.multiply_operators(['Sx','Sy'])
        site.multiply_operators(['Sy','Sx'])
        site.multiply_operators(['Sy','Sz'])
        site.multiply_operators(['Sz','Sy'])
        site.multiply_operators(['Sz','Sx'])
        site.multiply_operators(['Sx','Sz'])
               
        lat = Triangular( Lx=Lx, Ly=Ly, site=site, bc=bc, bc_MPS=bc_MPS )
        CouplingModel.__init__(self, lat)

        # Magnetic field
        self.add_onsite( -h, 0, 'Sz')

        # Easy-plane anisotropy
        self.add_onsite( D, 0, 'Sz Sz')
        
        # NN XXZ
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling(-J1, u1, 'Sx', u2, 'Sx', dx)
            self.add_coupling(-J1, u1, 'Sy', u2, 'Sy', dx)
            self.add_coupling(-J1*delta, u1, 'Sz', u2, 'Sz', dx)
        
        # NN XXZ
        for u1, u2, dx in self.lat.pairs['next_nearest_neighbors']:
            self.add_coupling(-J2, u1, 'Sx', u2, 'Sx', dx)
            self.add_coupling(-J2, u1, 'Sy', u2, 'Sy', dx)
            self.add_coupling(-J2*delta, u1, 'Sz', u2, 'Sz', dx)
        
        MPOModel.__init__(self, lat, self.calc_H_MPO())