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
                   'z_max': 4., # chose a wide range 
                   'redshift_epsrel': 1e-6,
                   'redshift_epsabs': 1e-100,  
    
                   'M_min': 1e13, # chose a wide range 
                   'M_max': 1e17, # chose a wide range 
                   'mass_epsrel':1e-6,
                   'mass_epsabs':1e-100,
   
                   'ndim_redshifts' :500,
                   'ndim_masses' : 500,
                   'n_m_dndlnM' : 500,
                   'n_z_dndlnM' : 500,
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
    def __init__(self, minMass, areaDeg2, zMin, zMax, H0, Om0, Ob0, sigma8, ns, zStep = 0.01, 
                 enableDrawSample = False, delta = 500, rhoType = 'critical', 
                 transferFunction = 'boltzmann_camb', massFunction = 'Tinker08',
                 c_m_relation = 'Bhattacharya13',scalingRelationDict = None):
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

        zRange=np.arange(zMin, zMax+zStep, zStep)
        self.zBinEdges=zRange
        self.z=(zRange[:-1]+zRange[1:])/2.
        self.a=1./(1+self.z)
        
        self.delta=delta
        self.rhoType=rhoType
        
        
        self.log10M=np.arange(np.log10(minMass), 16, 0.001)
        self.M=np.power(10, self.log10M)
        self.log10MBinEdges=np.linspace(self.log10M.min()-(self.log10M[1]-self.log10M[0])/2, 
                                        self.log10M.max()+(self.log10M[1]-self.log10M[0])/2, len(self.log10M)+1)  


        self.enableDrawSample = enableDrawSample
        
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


        self.cosmo = Class()
        self.cosmo.set(common_class_sz_settings)
        self.cosmo.set(class_sz_cosmo_params)
        self.cosmo.set(class_sz_ym_params)
        self.cosmo.compute_class_szfast()
        
        print('>>> class_sz computed')
        print('>>> updating cosmological quantities')
        self.update(H0, Om0, Ob0, sigma8, ns)


    def setSurveyArea(self, areaDeg2):
        """Change the area of the survey to a user-specified value, updating the cluster
        counts accordingly.

        Args:
            areaDeg2 (:obj:`float`): Area of the survey in square degrees.

        """

        if areaDeg2 == 0:
            raise Exception("Cannot create a MockSurvey object with zero area")
        areaSr=np.radians(np.sqrt(areaDeg2))**2
        if areaDeg2 != self.areaDeg2:
            self.areaSr=areaSr
            self.areaDeg2=areaDeg2
            self._doClusterCount()



            
    def update(self, H0, Om0, Ob0, sigma8, ns):
        """Recalculate cluster counts for the updated cosmological parameters given.
        
        Args:
            H0 (:obj:`float`): The Hubble constant at redshift 0, in km/s/Mpc.
            Om0 (:obj:`float`): Dimensionless total (dark + baryonic) matter density parameter at redshift 0.
            Ob0 (:obj:`float`): Dimensionless baryon matter density parameter at redshift 0.
            sigma8 (:obj:`float`): Defines the amplitude of the matter power spectrum.
            ns (:obj:`float`): Scalar spectral index of matter power spectrum.  
                
        """

        
        self._doClusterCount()
        
        # For quick Q calc (these are in MockSurvey rather than SelFn as used by drawSample)
        self.theta500Splines=[]

        self.Ez = np.vectorize(self.cosmo.Hubble)(self.z)/self.cosmo.Hubble(0.)
        self.Ez2 = np.power(self.Ez, 2)
        
        self.DAz = np.vectorize(self.cosmo.get_chi)(self.z)/(1.+self.z)/self.cosmo.h()
        
        self.criticalDensity = np.vectorize(self.cosmo.get_rho_crit_at_z)(self.z)*self.cosmo.h()**2
        
        for k in range(len(self.z)):
            
            
            interpLim_minLog10M = self.log10M.min() # this is m200c
            interpLim_maxLog10M = self.log10M.max() # this is m200c
            
            zk = self.z[k]
            # print(">>> tabulating a few things at z: ",zk)
            
            interpPoints = 100
            
            fitM500s = np.power(10, np.linspace(interpLim_minLog10M, interpLim_maxLog10M, interpPoints))
            
            fitM500s = np.vectorize(self.cosmo.get_m200c_to_m500c_at_z_and_M)(zk,fitM500s*self.cosmo.h())/self.cosmo.h() ## in Msun
            

            criticalDensity = self.criticalDensity[k]
            DA = self.DAz[k]
            Ez = self.Ez[k]
            
            R500Mpc = np.power((3*fitM500s)/(4*np.pi*500*criticalDensity), 1.0/3.0)    
            fitTheta500s = np.degrees(np.arctan(R500Mpc/DA))*60.0

            tckLog10MToTheta500 = interpolate.splrep(np.log10(fitM500s), fitTheta500s)
            self.theta500Splines.append(tckLog10MToTheta500)


        # Stuff to enable us to draw mock samples (see drawSample)
        # Interpolators here need to be updated each time we change cosmology
        if self.enableDrawSample == True:

            # For drawing from overall z distribution
            zSum = self.clusterCount.sum(axis = 1)
            pz = np.cumsum(zSum)/self.numClusters
            self.zRoller = _spline(pz, self.z, k = 3)
            
            # For drawing from each log10M distribution at each point on z grid
            # And quick fRel, Q calc using interpolation
            # And we may as well have E(z), DA on the z grid also
            self.log10MRollers = []
            
            for i in range(len(self.z)):
                
                ngtm = self._cumulativeNumberDensity(self.z[i])
                mask = ngtm > 0
                self.log10MRollers.append(_spline((ngtm[mask] / ngtm[0])[::-1], np.log10(self.M[mask][::-1]), k=3))
    

    def _cumulativeNumberDensity(self, z):
        """Returns N > M (per cubic Mpc).
        
        """

        h = self.cosmo.h()
        dndlnM = np.vectorize(self.cosmo.get_dndlnM_at_z_and_M)(z,self.M*self.cosmo.h())*self.cosmo.h()**3
        dndM = dndlnM/self.M
        ngtm = integrate.cumtrapz(dndlnM[::-1], np.log(self.M), initial = 0)[::-1]
        
        MUpper = np.arange(np.log(self.M[-1]), np.log(10**17), np.log(self.M[1])-np.log(self.M[0]))
        extrapolator = _spline(np.log(self.M), np.log(dndlnM), k=1)
        MF_extr = extrapolator(MUpper)
        intUpper = integrate.simps(np.exp(MF_extr), dx=MUpper[2] - MUpper[1], even='first')
        ngtm = ngtm + intUpper
    
        return ngtm
    
    
    def _comovingVolume(self, z):
        """Returns co-moving volume in Mpc^3 (all sky) to some redshift z.
                
        """
        return 4.18879020479 * (np.vectorize(self.cosmo.get_chi)(z)/self.cosmo.h())**3

        
    def _doClusterCount(self):
        """Updates cluster count etc. after mass function object is updated.
        
        """

        assert(self.areaSr == np.radians(np.sqrt(self.areaDeg2))**2)

        zRange=self.zBinEdges
        
        h = self.cosmo.h()
        
        self.M=np.power(10, self.log10M) # in M_sun
        
        #BB
        print('>>> fsky:',(self.areaSr/(4*np.pi)))

        # Number density by z and total cluster count (in redshift shells)
        # Can use to make P(m, z) plane
        numberDensity = []
        clusterCount = []
        totalVolumeMpc3 = 0.
        
        print(">>> starting z loop")
        
        for i in range(len(zRange)-1):
            
            zShellMin = zRange[i]
            zShellMax = zRange[i+1]
            zShellMid = (zShellMax+zShellMin)/2.
            
            # print(">>>> zShellMin, zShellMax: ",zShellMin,zShellMax)
            
            dndlnM = np.vectorize(self.cosmo.get_dndlnM_at_z_and_M)(zShellMid,self.M*self.cosmo.h())*self.cosmo.h()**3
 
            nzfine = 10
            zfine = np.linspace(zShellMin,zShellMax,nzfine)
            dNdlnM = np.zeros((nzfine,len(self.M)))
            dVfine =  np.zeros(nzfine)
            
            for (izz,zzfine) in enumerate(zfine):
                

                dndlnMzfine = np.vectorize(self.cosmo.get_dndlnM_at_z_and_M)(zzfine,self.M*self.cosmo.h())*self.cosmo.h()**3
                
                Ezfine = np.vectorize(self.cosmo.Hubble)(zzfine)/self.cosmo.Hubble(0.)
                DAzfine = np.vectorize(self.cosmo.get_chi)(zzfine)/(1.+zzfine)/self.cosmo.h()
                       
                
                dNdlnM[izz,:] = 4.*np.pi*(self.areaSr/(4*np.pi))*dndlnMzfine*(1.+zzfine)**2*DAzfine**2/Ezfine*2.99792458e8/1e5/self.cosmo.h()
                dVfine[izz] = 4.*np.pi*(self.areaSr/(4*np.pi))*(1.+zzfine)**2*DAzfine**2/Ezfine*2.99792458e8/1e5/self.cosmo.h()
            
            dNdlnM = np.trapz(dNdlnM,x=zfine,axis=0)
            shellvolumefine = np.trapz(dVfine,x=zfine)
            nfine = dNdlnM/self.M*np.gradient(self.M)
            dndM = dndlnM / self.M
            n = dndM * np.gradient(self.M)
            
            numberDensity.append(nfine/shellvolumefine)
            
            shellVolumeMpc3 = self._comovingVolume(zShellMax)-self._comovingVolume(zShellMin)
            shellVolumeMpc3 = shellVolumeMpc3*(self.areaSr/(4*np.pi))
            totalVolumeMpc3 += shellvolumefine
            
            clusterCount.append(nfine)
            # print(">>> nfine: ",nfine)
            
        numberDensity = np.array(numberDensity)
        clusterCount = np.array(clusterCount)
        
        self.volumeMpc3 = totalVolumeMpc3
        
        self.numberDensity = numberDensity
        
        self.clusterCount = clusterCount
        self.numClusters = np.sum(clusterCount)
        
        self.numClustersByRedshift = np.sum(clusterCount, axis = 1)
        print(">>> doCluster done")


    def calcNumClustersExpected(self, MLimit = 1e13, zMin = 0.0, zMax = 4.0, compMz = None):
        
        return 0
        


    def drawSample(self, 
                   y0Noise, 
                   scalingRelationDict, 
                   QFit = None, 
                   wcs = None, 
                   photFilterLabel = None,
                   tileName = None, 
                   SNRLimit = None, 
                   makeNames = False, 
                   z = None, 
                   numDraws = None,
                   areaDeg2 = None, 
                   applySNRCut = False, 
                   applyPoissonScatter = True,
                   applyIntrinsicScatter = True, 
                   applyNoiseScatter = True,
                   applyRelativisticCorrection = True, 
                   verbose = False, 
                   biasModel = None):
        """Draw a cluster sample from the mass function, generating mock y0~ values (called `fixed_y_c` in
        Nemo catalogs) by applying the given scaling relation parameters, and then (optionally) applying
        a survey selection function.
        
        Args:
            y0Noise (:obj:`float` or :obj:`np.ndarray`): Either a single number (if using e.g., a survey
                average), an RMS table (with columns 'areaDeg2' and 'y0RMS'), or a noise map (2d array).
                A noise map must be provided here if you want the output catalog to contain RA, dec
                coordinates (in addition, a WCS object must also be provided - see below).
            scalingRelationDict (:obj:`dict`): A dictionary containing keys 'tenToA0', 'B0', 'Mpivot',
                'sigma_int' that describes the scaling relation between y0~ and mass (this is the
                format of `massOptions` in Nemo .yml config files).
            QFit (:obj:`nemo.signals.QFit`, optional): Object that handles the filter mismatch
                function, *Q*. If not given, the output catalog will not contain `fixed_y_c` columns,
                only `true_y_c` columns.
            wcs (:obj:`astWCS.WCS`, optional): WCS object corresponding to `y0Noise`, if `y0Noise` is
                as noise map (2d image array). Needed if you want the output catalog to contain RA, dec
                coordinates.
            photFilterLabel (:obj:`str`, optional): Name of the reference filter (as defined in the
                Nemo .yml config file) that is used to define y0~ (`fixed_y_c`) and the filter mismatch 
                function, Q.
            tileName (:obj:`str`, optional): Name of the tile for which the sample will be generated.
            SNRLimit (:obj:`float`, optional): Signal-to-noise detection threshold used for the
                output catalog (corresponding to a cut on `fixed_SNR` in Nemo catalogs). Only applied
                if `applySNRCut` is also True (yes, this can be cleaned up).
            makeNames (:obj:`bool`, optional): If True, add names of the form MOCK CL JHHMM.m+/-DDMM
                to the output catalog.
            z (:obj:`float`, optional): If given produce a sample at the nearest z in the MockSurvey
                z grid. The default behaviour is to use the full redshift grid specified by `self.z`.
            numDraws (:obj:`int`, optional): If given, the number of draws to perform from the mass
                function, divided equally among the redshift bins. The default is to use the values
                contained in `self.numClustersByRedshift`.
            areaDeg2 (:obj:`float`, optional): If given, the cluster counts will be scaled to this
                area. Otherwise, they correspond to `self.areaDeg2`. This parameter will be ignored
                if `numDraws` is also given.
            applySNRCut (:obj:`bool`, optional): If True, cut the output catalog according to the
                `fixed_SNR` threshold set by `SNRLimit`.
            applyPoissonScatter (:obj:`bool`, optional): If True, add Poisson noise to the cluster
                counts (implemented by modifiying the number of draws from the mass function).
            applyIntrinsicScatter (:obj:`bool`, optional): If True, apply intrinsic scatter to the
                SZ measurements (`fixed_y_c`), as set by the `sigma_int` parameter in 
                `scalingRelationDict`.
            applyNoiseScatter (:obj:`bool`, optional): If True, apply measurement noise, generated
                from the given noise level or noise map (`y0Noise`), to the output SZ measurements
                (`fixed_y_c`).
            applyRelativisticCorrection (:obj:`bool`, optional): If True, apply the relativistic
                correction.
                
        Returns:
            A catalog as an :obj:`astropy.table.Table` object, in the same format as produced by
            the main `nemo` script.
        
        Notes:
            If both `applyIntrinsicScatter`, `applyNoiseScatter` are set to False, then the output
            catalog `fixed_y_c` values will be exactly the same as `true_y_c`, although each object
            will still have an error bar listed in the output catalog, corresponding to its location
            in the noise map (if given).
                
        """

        t0=time.time()
        if z is None:
            zRange = self.z
        else:
            # Pick the nearest z on the grid
            zIndex = np.argmin(abs(z-self.z))
            zRange = [self.z[zIndex]]
        
        # Add Poisson noise (we do by z to keep things simple on the z grid later)
        numClustersByRedshift=np.zeros(len(zRange), dtype = int)
        
        for k in range(len(zRange)):
            zk=zRange[k]
            zIndex=np.argmin(abs(zk-self.z))
            if applyPoissonScatter == False:
                numClustersByRedshift[k]=int(round(self.numClustersByRedshift[zIndex]))
            else:
                numClustersByRedshift[k]=np.random.poisson(int(round(self.numClustersByRedshift[zIndex])))

        if areaDeg2 is not None:
            numClustersByRedshift=np.array(numClustersByRedshift*(areaDeg2/self.areaDeg2), dtype = int)
        
        numClusters=numClustersByRedshift.sum()
            
        if numDraws is not None:
            numClusters=numDraws            

        # If given y0Noise as RMSMap, draw coords (assuming clusters aren't clustered - which they are...)
        # NOTE: switched to using valid part of RMSMap here rather than areaMask - we need to fix the latter to same area
        # It isn't a significant issue though
        if type(y0Noise) == np.ndarray and y0Noise.ndim == 2:
            # This generates even density RA, dec coords on the whole sky taking into account the projection
            # Consequently, this is inefficient if fed individual tiles rather than a full sky noise map
            assert(wcs is not None)
            RMSMap=y0Noise
            xsList=[]
            ysList=[]
            maxCount=10000
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
        elif type(y0Noise) == atpy.Table:
            noisetck=interpolate.splrep(np.cumsum(y0Noise['areaDeg2']/y0Noise['areaDeg2'].sum()), y0Noise['y0RMS'], k = 1)
            rnd=np.random.uniform(0, 1, numClusters)
            vals=interpolate.splev(rnd, noisetck, ext = 3)
            if np.any(vals < 0) or np.any(vals == np.nan):
                raise Exception("Failed to make interpolating spline for RMSTab in tileName = %s" % (tileName))
            y0Noise=vals
            RAs=np.zeros(numClusters)
            decs=np.zeros(numClusters)
        else:
            y0Noise=np.ones(numClusters)*y0Noise
            RAs=np.zeros(numClusters)
            decs=np.zeros(numClusters)
            
        
        
        # y0Noise=np.ones(numClusters)*y0Noise
        # RAs=np.zeros(numClusters)
        # decs=np.zeros(numClusters)
        
        # Fancy names or not?
        if makeNames == True:
            names=[]
            for RADeg, decDeg in zip(RAs, decs):
                names.append(catalogs.makeName(RADeg, decDeg, prefix = 'MOCK-CL'))
        else:
            names=np.arange(numClusters)+1
                
        # New way - on the redshift grid
        t0 = time.time()
        
        currentIndex = 0
        
        log10Ms = np.random.random_sample(y0Noise.shape) # These will be converted from random numbers to masses below
        log10M500cs = np.zeros(y0Noise.shape)
        
        zs = np.zeros(y0Noise.shape)
        zErrs = np.zeros(y0Noise.shape)
        
        Ez2s = np.zeros(y0Noise.shape)


        if verbose:
            print(">>> Generating mock sample in redshift slices")
        for k in range(len(zRange)):

            # if verbose:
            print("... z = %.3f [%d/%d]" % (zRange[k], k+1, len(zRange)))
            
            t00=time.time()
            
            zk=zRange[k]
            zIndex=np.argmin(abs(zk-self.z))      
            
            if numDraws is not None:
                numClusters_zk = int(round(numDraws/len(zRange)))
                print(">> numDraws/len(zRange): ",numDraws/len(zRange),numDraws)
            
            else:
                numClusters_zk = numClustersByRedshift[k]
                print(">> numClustersByRedshift[k]: ",numClustersByRedshift[k])
            
            if numClusters_zk == 0:
                continue
            
            t11=time.time()
            
            # Some fiddling here to avoid rounding issues with array sizes (+/-1 draw here shouldn't make a difference)
            nextIndex = currentIndex + numClusters_zk
            
            if nextIndex >= len(y0Noise):
                nextIndex=len(y0Noise)
            
            mask=np.arange(currentIndex, nextIndex)
            
            numClusters_zk=len(mask)
            if numClusters_zk == 0:
                continue
            
            currentIndex = nextIndex
            
            t22=time.time()
            
            log10Ms[mask] = self.log10MRollers[k](log10Ms[mask])
            
            log10M500cs[mask] = np.log10(np.vectorize(self.cosmo.get_m200c_to_m500c_at_z_and_M)(zk,10**log10Ms[mask]*self.cosmo.h())/self.cosmo.h()) 
            
            theta500s=interpolate.splev(log10M500cs[mask], self.theta500Splines[k], ext = 3)
            
                
            Ez2s[mask]=self.Ez2[k]
            zs[mask]=zk
      
        # For some cosmo parameters, splined masses can end up outside of valid range, so catch this
        log10Ms[log10Ms < self.log10M.min()]=self.log10M.min()
        log10Ms[log10Ms > self.log10M.max()]=self.log10M.max()
        

        M200c = np.power(10, log10Ms)

        true_y0s = np.vectorize(self.cosmo.get_y_at_m_and_z)(M200c*self.cosmo.h(),zs)
            
            
        # Add noise and intrinsic scatter everywhere
        if applyIntrinsicScatter == True:
            
            scattered_y0s = np.exp(np.random.normal(np.log(true_y0s), sigma_int, len(true_y0s)))
            
        else:
            
            scattered_y0s = true_y0s
            
        if applyNoiseScatter == True:
            
            measured_y0s = np.random.normal(scattered_y0s, y0Noise)
        
        else:
            
            measured_y0s = scattered_y0s
        

        massColLabel = "true_M200c"
        tab = atpy.Table()
        
        tab.add_column(atpy.Column(names, 'name'))
        tab.add_column(atpy.Column(RAs, 'RADeg'))
        tab.add_column(atpy.Column(decs, 'decDeg'))
        tab.add_column(atpy.Column(np.power(10, log10Ms)/1e14, massColLabel))
        
        if 'true_M500c' not in tab.keys():
            
            tab.add_column(atpy.Column(np.power(10, log10M500cs)/1e14, 'true_M500c'))
        
        if QFit is None:
        
            tab.add_column(atpy.Column(true_y0s/1e-4, 'true_y_c'))
        
        else:
            
            tab.add_column(atpy.Column(Qs, 'true_Q'))
            tab.add_column(atpy.Column(true_y0s/1e-4, 'true_fixed_y_c'))
            tab.add_column(atpy.Column(measured_y0s/1e-4, 'fixed_y_c'))

        tab.add_column(atpy.Column(zs, 'redshift'))
        tab.add_column(atpy.Column(zErrs, 'redshiftErr'))
        
        if photFilterLabel is not None and tileName is not None:
        
            tab.add_column(atpy.Column([photFilterLabel]*len(tab), 'template'))
            tab.add_column(atpy.Column([tileName]*len(tab), 'tileName'))
                
        t1=time.time()

        return tab
