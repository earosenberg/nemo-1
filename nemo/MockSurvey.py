"""

This module defines the MockSurvey class, used for mass function calculations, obtaining de-biased cluster
mass estimates, selection function calculations, and generating mock catalogs.

"""

import os
import sys
import numpy as np
import astropy.table as atpy
import pylab as plt
import subprocess
from astropy.cosmology import FlatLambdaCDM
on_rtd=os.environ.get('READTHEDOCS', None)
if on_rtd is None:
    import pyccl as ccl
from . import signals
from . import catalogs
from . import maps
import pickle
from scipy import interpolate
from scipy import integrate
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy import stats
from astLib import *
import time

from classy_sz import Class

common_class_sz_settings = {
                   'mass function' : 'T08M200c', 
                   'hm_consistency': 0,
                   'concentration parameter' : 'B13',
                   'B':1.,

                   'N_ncdm' : 1,
                   'N_ur' : 2.0328,
                   'm_ncdm' : 0.0,
                   'T_ncdm' : 0.71611,
    
                   'z_min': 1.e-3, # chose a wide range 
                   'z_max': 3.05, # chose a wide range 
                   'redshift_epsrel': 1e-6,
                   'redshift_epsabs': 1e-100,  
    
                   'M_min': 1e13, # chose a wide range 
                   'M_max': 1e17, # chose a wide range 
                   'mass_epsrel':1e-6,
                   'mass_epsabs':1e-100,
   
                   'ndim_redshifts' :5000,
                   'ndim_masses' : 500,
                   'n_m_dndlnM' : 1000,
                   'n_z_dndlnM' : 1000,
                   'HMF_prescription_NCDM': 1,
                   'no_spline_in_tinker': 1,
    
    
                   'use_m500c_in_ym_relation' : 0,
                   'use_m200c_in_ym_relation' : 1,
                   'y_m_relation' : 1,
    
                   'output': 'dndlnM,m500c_to_m200c,m200c_to_m500c',
}


#------------------------------------------------------------------------------------------------------------
class MockSurvey(object):
    """An object that provides routines calculating cluster counts (using `CCL <https://ccl.readthedocs.io/en/latest/>`_) 
    and generating mock catalogs for a given set of cosmological and mass scaling relation parameters.
    The Tinker et al. (2008) halo mass function is used (hardcoded at present, but in principle this can
    easily be swapped for any halo mass function supported by CCL).
    
    Attributes:
        areaDeg2 (:obj:`float`): Survey area in square degrees.
        zBinEdges (:obj:`np.ndarray`): Defines the redshift bins for the cluster counts.
        z (:obj:`np.ndarray`): Centers of the redshift bins.
        log10M (:obj:`np.ndarray`): Centers of the log10 mass bins for the cluster counts
            (in MSun, with mass defined according to `delta` and `rhoType`).
        a (:obj:`np.ndarray`): Scale factor (1/(1+z)).
        delta (:obj:``float`): Overdensity parameter, used for mass definition (e.g., 200, 500).
        rhoType (:obj:`str`): Density definition, either 'matter' or 'critical', used for mass definition.
        mdef (:obj:`pyccl.halos.massdef.MassDef`): CCL mass definition object, defined by `delta` and `rhoType`.
        transferFunction (:obj:`str`): Transfer function to use, as understood by CCL (e.g., 'eisenstein_hu', 
            'boltzmann_camb').
        H0 (:obj:`float`): The Hubble constant at redshift 0, in km/s/Mpc.
        Om0 (:obj:`float`): Dimensionless total (dark + baryonic) matter density parameter at redshift 0.
        Ob0 (:obj:`float`): Dimensionless baryon matter density parameter at redshift 0.
        sigma8 (:obj:`float`): Defines the amplitude of the matter power spectrum.
        ns (:obj:`float`): Scalar spectral index of matter power spectrum.
        volumeMpc3 (:obj:`float`): Co-moving volume in Mpc3 for the given survey area and cosmological
            parameters.
        numberDensity (:obj:`np.ndarray`): Number density of clusters (per cubic Mpc) on the 
            (z, log10M) grid.
        clusterCount (:obj:`np.ndarray`): Cluster counts on the (z, log10M) grid.
        numClusters (:obj:`float`): Total number of clusters in the survey area above the minimum mass
            limit.
        numClustersByRedshift (:obj:`np.ndarray`): Number of clusters in the survey area above the
            minimum mass limit, as a function of redshift.
    
    """
    def __init__(self, 
                 minMass, 
                 maxMass,
                 areaDeg2, 
                 zMin, 
                 zMax, 
                 H0, Om0, Ob0, sigma8, ns,
                 scalingRelationDict = None):
        """Create a MockSurvey object, for performing calculations of cluster counts or generating mock
        catalogs. The Tinker et al. (2008) halo mass function is used (hardcoded at present, but in 
        principle this can easily be swapped for any halo mass function supported by CCL).
        
        Args:
            minMass (:obj:`float`): The minimum mass, in MSun. This should be set considerably lower than
                the actual survey completeness limit, otherwise completeness calculations will be wrong.
            areaDeg2 (:obj:`float`): Specifies the survey area in square degrees, which scales the
                resulting cluster counts accordingly.
            zMin (:obj:`float`): Minimum redshift for the (z, log10M) grid.
            zMax (:obj:`float`): Maximum redshift for the (z, log10M) grid.
            H0 (:obj:`float`): The Hubble constant at redshift 0, in km/s/Mpc.
            Om0 (:obj:`float`): Dimensionless total (dark + baryonic) matter density parameter at redshift 0.
            Ob0 (:obj:`float`): Dimensionless baryon matter density parameter at redshift 0.
            sigma8 (:obj:`float`): Defines the amplitude of the matter power spectrum.
            ns (:obj:`float`): Scalar spectral index of matter power spectrum.  
            zStep (:obj:`float`, optional): Sets the linear spacing between redshift bins.
            enableDrawSample (:obj:`bool`, optional): This needs to be set to True to enable use of the
                :func:`self.drawSample` function. Setting this to False avoids some overhead.
            delta (:obj:``float`): Overdensity parameter, used for mass definition (e.g., 200, 500).
            rhoType (:obj:`str`): Density definition, either 'matter' or 'critical', used for mass definition.
            transferFunction (:obj:`str`): Transfer function to use, as understood by CCL (e.g., 'eisenstein_hu', 
                'boltzmann_camb').
            massFunction (:obj:`str`): Name of the mass function to use, currently either 'Tinker08' or
                'Tinker10'. Mass function calculations are done by CCL.
            c_m_relation ('obj':`str`): Name of the concentration -- mass relation to assume, as understood by
                CCL (this may be used internally for conversion between mass definitions, as needed).

        """
        
        if areaDeg2 == 0:
            raise Exception("Cannot create a MockSurvey object with zero area")
        self.areaDeg2=areaDeg2
        self.areaSr=np.radians(np.sqrt(areaDeg2))**2

        
        self.log10M=np.arange(np.log10(minMass), np.log10(maxMass), 0.001)

        
        class_sz_cosmo_params = {
        'Omega_b': Ob0,
        'Omega_cdm':  Om0-Ob0,
        'H0': H0,
        'sigma8': sigma8,
        'tau_reio':  0.0561, ## doesnt matter 
        'n_s': ns,
        }
        
        tenToA0, B0, Mpivot, sigma_int = [scalingRelationDict['tenToA0'], 
                                          scalingRelationDict['B0'], 
                                          scalingRelationDict['Mpivot'], 
                                          scalingRelationDict['sigma_int']]
        
        class_sz_ym_params = {
        'A_ym'  : tenToA0,
        'B_ym'  : B0,
        'C_ym' : 0.,
        'sigmaM_ym' : sigma_int,
        'm_pivot_ym_[Msun]' : Mpivot,   
        }
        #BB


        self.cosmo = Class()
        self.cosmo.set(common_class_sz_settings)
        self.cosmo.set(class_sz_cosmo_params)
        self.cosmo.set(class_sz_ym_params)
        self.cosmo.compute_class_szfast()
        
        print('>>> computing ntot from hmf')
        self.fsky = self.areaSr/(4*np.pi)
        print('>>> fsky:',self.fsky)
        

        z_min = zMin 
        z_max = zMax 

        m200_min = minMass*self.cosmo.h() 
        m200_max = maxMass*self.cosmo.h() 

        nms = 20000 # default : 20000
        nzs = 10000 # default : 10000

        lnms = np.linspace(np.log(m200_min),np.log(m200_max),nms)
        zs = np.linspace(z_min,z_max,nzs)
        
        self.z = zs
        
        dndmdz = np.zeros((nms,nzs))

        for (im,mm) in enumerate(lnms):
            dndmdz[im,:] = 4.*np.pi*self.fsky*np.vectorize(self.cosmo.get_volume_dVdzdOmega_at_z)(zs)*np.vectorize(self.cosmo.get_dndlnM_at_z_and_M)(zs,np.exp(mm))
        
        self.hmf_range = np.array([np.min(dndmdz),np.max(dndmdz)])

        # integrate over massses at each z:
        dndz = np.trapz(dndmdz,x=lnms,axis=0)
        th_ntot = np.trapz(dndz,x = zs)
        
        print('>>> class_sz computed')
        print('>>> got ntot: ', th_ntot)
        
        self.n_tot_obs = np.random.poisson(lam=th_ntot)
        
        print('>>> ntot obs for this catalog: ', self.n_tot_obs)


    def setSurveyArea(self, areaDeg2):
        """Change the area of the survey to a user-specified value, updating the cluster
        counts accordingly.

        Args:
            areaDeg2 (:obj:`float`): Area of the survey in square degrees.

        """
        return 0 



            
    def update(self, H0, Om0, Ob0, sigma8, ns):
        """Recalculate cluster counts for the updated cosmological parameters given.
        
        Args:
            H0 (:obj:`float`): The Hubble constant at redshift 0, in km/s/Mpc.
            Om0 (:obj:`float`): Dimensionless total (dark + baryonic) matter density parameter at redshift 0.
            Ob0 (:obj:`float`): Dimensionless baryon matter density parameter at redshift 0.
            sigma8 (:obj:`float`): Defines the amplitude of the matter power spectrum.
            ns (:obj:`float`): Scalar spectral index of matter power spectrum.  
                
        """
        return 0 

    def _cumulativeNumberDensity(self, z):
        """Returns N > M (per cubic Mpc).
        
        """
    
        return 0
    
    
    def _comovingVolume(self, z):
        """Returns co-moving volume in Mpc^3 (all sky) to some redshift z.
                
        """
        return 0

        
    def _doClusterCount(self):
        """Updates cluster count etc. after mass function object is updated.
        
        """
        return 0 


    def calcNumClustersExpected(self, MLimit = 1e13, zMin = 0.0, zMax = 4.0, compMz = None):
        
        return 0
        


    def drawSample(self, 
                   y0Noise, 
                   scalingRelationDict, 
                   wcs = None, 
                   numDraws = None):
        """Draw a cluster sample from the mass function, generating mock y0~ values (called `fixed_y_c` in
        Nemo catalogs) by applying the given scaling relation parameters, and then (optionally) applying
        a survey selection function.
        """
        numClusters=numDraws            

        # If given y0Noise as RMSMap, draw coords (assuming clusters aren't clustered - which they are...)
        # NOTE: switched to using valid part of RMSMap here rather than areaMask - we need to fix the latter to same area
        # It isn't a significant issue though

        # This generates even density RA, dec coords on the whole sky taking into account the projection
        # Consequently, this is inefficient if fed individual tiles rather than a full sky noise map
        print(">>> even coord case")

        RMSMap=y0Noise
        xsList=[]
        ysList=[]
        maxCount=100000
        count=0
        while(len(xsList) < numClusters):
            count=count+1
            if count > maxCount:
                raise Exception("Failed to generate enough random coords in %d iterations" % (maxCount))
            theta=np.degrees(np.pi*2*np.random.uniform(0, 1, numClusters))
            phi=np.degrees(np.arccos(2*np.random.uniform(0, 1, numClusters)-1))-90
            xyCoords=np.array(wcs.wcs2pix(theta, phi))
            xs=np.array(np.round(xyCoords[:, 0]), dtype = int)
            ys=np.array(np.round(xyCoords[:, 1]), dtype = int)
            mask=np.logical_and(np.logical_and(xs >= 0, xs < RMSMap.shape[1]), np.logical_and(ys >= 0, ys < RMSMap.shape[0]))
            xs=xs[mask]
            ys=ys[mask]
            mask=RMSMap[ys, xs] > 0
            xsList=xsList+xs[mask].tolist()
            ysList=ysList+ys[mask].tolist()
        xs=np.array(xsList)[:numClusters]
        ys=np.array(ysList)[:numClusters]
        del xsList, ysList

        RADecCoords=wcs.pix2wcs(xs, ys)
        RADecCoords=np.array(RADecCoords)
        
        
        RAs=RADecCoords[:, 0]
        decs=RADecCoords[:, 1]
        y0Noise=RMSMap[ys, xs]

        ## now draw a mass and a redshift value
        zs = np.zeros(y0Noise.shape)
        zErrs = np.zeros(y0Noise.shape)
        log10Ms = np.zeros(y0Noise.shape)
        
        ## credit: Inigo Zubeldia 
        print(">>> rejection sampling (from Inigo Zubeldia)")
        for cluster_index in range(numClusters):
            hmf_eval = 0.
            hmf_sample = 1.
            
            # rejection sampling 
            while hmf_sample > hmf_eval:
                ln10m_sample = np.random.rand()*(self.log10M[-1]-self.log10M[0])+self.log10M[0]
                lnmh_sample = np.log(10**ln10m_sample*self.cosmo.h())
                z_sample = np.random.rand()*(self.z[-1]-self.z[0])+self.z[0]
                hmf_sample = np.random.rand()*(self.hmf_range[-1]-self.hmf_range[0])+self.hmf_range[0]
                hmf_eval = 4.*np.pi*self.fsky*self.cosmo.get_volume_dVdzdOmega_at_z(z_sample)*self.cosmo.get_dndlnM_at_z_and_M(z_sample,np.exp(lnmh_sample))
                
            log10Ms[cluster_index] = ln10m_sample
            zs[cluster_index] = z_sample
        
                   

        names=[]
        for RADeg, decDeg in zip(RAs, decs):
            names.append(catalogs.makeName(RADeg, decDeg, prefix = 'MOCK-CL'))
                
        
        
        M200c = np.power(10, log10Ms)
        true_y0s = np.vectorize(self.cosmo.get_y_at_m_and_z)(M200c*self.cosmo.h(),zs)
        
        tab = atpy.Table()
        
        tab.add_column(atpy.Column(names, 'name'))
        tab.add_column(atpy.Column(RAs, 'RADeg'))
        tab.add_column(atpy.Column(decs, 'decDeg'))
        tab.add_column(atpy.Column(np.power(10, log10Ms)/1e14, "true_M200c"))
        tab.add_column(atpy.Column(true_y0s/1e-4, 'true_y_c'))
        tab.add_column(atpy.Column(zs, 'redshift'))
        tab.add_column(atpy.Column(zErrs, 'redshiftErr'))
                

        return tab
