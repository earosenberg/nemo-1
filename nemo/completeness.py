"""

This module contains tools for calculating the completeness of cluster samples and handling area masks.

"""

import os
import sys
import resource
import glob
import numpy as np
import pylab as plt
import astropy.table as atpy
from astLib import *
from scipy import stats
from scipy import interpolate
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy import ndimage
from scipy import optimize
import nemo
from . import signals
from . import maps
from . import MockSurvey
from . import plotSettings
from . import startUp
from collections import OrderedDict
import colorcet
import types
import pickle
import astropy.io.fits as pyfits
import time
import shutil
import yaml
from decimal import Decimal

# If want to catch warnings as errors...
#import warnings
#warnings.filterwarnings('error')

#------------------------------------------------------------------------------------------------------------
class SelFn(object):
    """An object that describes the survey selection function. It uses the output in the ``selFn/`` directory
    (produced by the :ref:`nemoCommand` command) to calculate the survey completeness for a given
    signal-to-noise cut on a (log\ :sub:`10` mass, z) grid for a given set of cosmological and scaling
    relation parameters.
       
    Args:
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        SNRCut (:obj:`float`): Completeness will be computed relative to this signal-to-noise selection
            cut (labelled as `fixed_SNR` in the catalogs produced by :ref:`nemoCommand`).
        configFileName (:obj:`str`, optional): Path to a Nemo configuration file. If not given, this
            will be read from ``selFnDir/config.yml``, which is the config file for the 
            :ref:`nemoCommand` run that produced the ``selFn`` directory.
        footprintLabel (:obj:`str`, optional): Use this to specify a footprint, if any are defined in the
            Nemo config used to produce the ``selFn`` dir (e.g., 'DES', 'HSC', 'KiDS' etc.). The default
            value of ``None`` uses the whole survey footprint.
        zStep (:obj:`float`, optional): Use this to set the binning in redshift for completeness
            calculations.
        tileNames (:obj:`list`, optional): If given, restrict the :class:`SelFn` object to use only these
            tiles.
        enableDrawSample (:obj:`bool`, optional): This only needs to be set to `True` for generating mock
            catalogs.
        mockOversampleFactor (:obj:`float`, optional): Used only by :func:`generateMockSample`. Sets
            the oversampling level for the generated mock sample.
        downsampleRMS (:obj:`float`, optional): Downsample the resolution of the RMS (noise) tables by
            this factor. The RMS tables are generated from the noise maps, and are just a listing of noise
            level versus survey area. Downsampling speeds up completeness calculations considerably.
        applyMFDebiasCorrection (:obj:`bool`, optional): Set to `False` to disable the Eddington bias
            correction of mass estimates. Probably only useful for debugging.
        applyRelativisticCorrection (:obj:`bool`, optional): Set to `False` to disable inclusion of
            the relativistic correction in completeness calculations.
        setupAreaMask (:obj:`bool`, optional): If `True`, read in the area masks so that quick position
            checks can be done (e.g., by :meth:`SelFn.checkCoordsAreInMask`).
        enableCompletenessCalc (:obj:`bool`, optional): If `True`, set up the machinery needed to do
            completeness calculations.
        massFunction (:obj:`str`, optional): Name of the mass function to use, currently either
            'Tinker08' or 'Tinker10'. Mass function calculations are done by CCL.
        maxTheta500Arcmin (:obj:`float`, optional): If given, exclude clusters with expected angular size
            greater than this from the cluster counts (set their completeness to 0).
        method (:obj:`str, optional): The method for calculating completeness. Options are: 'fast'
            (using a simple model based on the noise and the expected cluster signals), 'injection'
            (directly using the results of end-to-end cluster injection and recovery sims).

    Attributes:
        SNRCut (:obj:`float`): Completeness will be computed relative to this signal-to-noise selection cut
            (labelled as `fixed_SNR` in the catalogs produced by :ref:`nemoCommand`).
        footprintLabel (:obj:`str`): Use this to specify a footprint, if any are defined in the Nemo config
            used to produce the ``selFn`` dir (e.g., 'DES', 'HSC', 'KiDS' etc.). The default of ``None``
            uses the whole survey footprint.
        applyMFDebiasCorrection (:obj:`bool`): Set to `False` to disable the Eddington bias correction of
            mass estimates. Probably only useful for debugging.
        zStep (:obj:`float`): Use this to set the binning in redshift for completeness calculations.
        zMax (:obj:`float`): Sets the maximum z used for completeness calculations.
        tileNames (:obj:`list`): The list of tiles used by the SelFn object (default of None uses all tiles).
        WCSDict (:obj:`dict`): A dictionary indexed by `tileName`, containing :obj:`astWCS.WCS` objects
            that describe the mapping between pixel coords and (RA, dec) coords in each tile.
        areaMaskDict (:obj:`dict`): A dictionary containing the survey area masks, indexed by tileName.
            Values > 0 in these masks define the cluster or source search area.
        scalingRelationDict (:obj:`dict`): A dictionary of scaling relation parameters (see example Nemo
            config files for the format).
        Q (:class:`nemo.signals.QFit`): An object for calculating the filter mismatch function, referred
            to as `Q` in the ACT papers from `Hasselfield et al. (2013) <http://adsabs.harvard.edu/abs/2013JCAP...07..008H>`_
            onwards.
        RMSDict (:obj:`dict`): A dictionary of RMS tables, indexed by tileName. Each RMSTable contains 
            the noise level by area, as returned by :meth:`getRMSTab`.
        totalAreaDeg2 (:obj:`float`): The total area in square degrees, as measured from the survey mask,
            for the given set of tiles and footprint.
        fRelDict (:obj:`dict`): A dictionary of weights used for relativistic corrections, indexed by
            `tileName`.
        mockSurvey (:class:`nemo.MockSurvey.MockSurvey`): A :class:`MockSurvey` object, used for halo mass function 
            calculations and generating mock catalogs.

    Note: 
        Some of the methods of this class are experimental and not necessarily well tested.
            
    """
        
    def __init__(self, selFnDir, SNRCut, configFileName = None, footprintLabel = None, zStep = 0.01,
                 zMax = 3.0, tileNames = None, enableDrawSample = False, mockOversampleFactor = 1.0,
                 downsampleRMS = True, applyMFDebiasCorrection = True, applyRelativisticCorrection = True,
                 setUpAreaMask = False, enableCompletenessCalc = True, delta = 500, rhoType = 'critical',
                 massFunction = 'Tinker08', maxTheta500Arcmin = None, method = 'fast'):
        
        self.SNRCut=SNRCut
        self.footprintLabel=footprintLabel
        self.downsampleRMS=downsampleRMS
        self.applyMFDebiasCorrection=applyMFDebiasCorrection
        self.applyRelativisticCorrection=applyRelativisticCorrection
        self.selFnDir=selFnDir
        self.zStep=zStep
        self.maxTheta500Arcmin=maxTheta500Arcmin

        if configFileName is None:
            configFileName=self.selFnDir+os.path.sep+"config.yml"
            if os.path.exists(configFileName) == False:
                raise Exception("No config .yml file found in selFnDir and no other location given.")
        self._config=startUp.NemoConfig(configFileName, makeOutputDirs = False, setUpMaps = False, verbose = False,
                                        selFnDir = self.selFnDir)
        parDict=self._config.parDict

        if tileNames is not None:
            self.tileNames=tileNames
        else:
            self.tileNames=self._config.tileNames
        
        self.method=method
        if self.method == 'injection':
            injDataPath=self.selFnDir+os.path.sep+"sourceInjectionData.fits"
            if os.path.exists(injDataPath) == True:
                self.injTab=atpy.Table().read(injDataPath)
            else:
                raise Exception("Cannot use the 'injection' method as %s does not exist." % (injDataPath))

        # Needed for generating mock samples directly
        self.photFilterLabel=self._config.parDict['photFilter']
            
        # Check that any given footprint is defined - if not, give a useful error message
        if footprintLabel is not None:
            if 'selFnFootprints' not in parDict.keys():
                raise Exception("No footprints defined in .yml config file")
            else:
                labelsList=[]
                for footprintDict in parDict['selFnFootprints']:
                    labelsList.append(footprintDict['label'])
                if footprintLabel not in labelsList:
                    raise Exception("Footprint '%s' not found in selFnFootprints - check .yml config file" % (footprintLabel))
        
        # Load area masks
        if setUpAreaMask == True:
            self._setUpAreaMask()
        else:
            self.tileTab=None
            self.WCSDict=None
            self.areaMaskDict=None
                
        if enableCompletenessCalc == True:
            
            self.scalingRelationDict=parDict['massOptions']

            # If the config didn't give cosmological parameters, put in defaults
            defaults={'H0': 70.0, 'Om0': 0.30, 'Ob0': 0.05, 'sigma8': 0.8, 'ns': 0.95}
            for key in defaults:
                if key not in self.scalingRelationDict.keys():
                    self.scalingRelationDict[key]=defaults[key]

            self.Q=signals.QFit(self.selFnDir+os.path.sep+"QFit.fits", tileNames = tileNames)
            
            # We should be able to do everything (except clustering) with this
            # NOTE: Some tiles may be empty, so we'll exclude them from tileNames list here
            RMSTabFileName=self.selFnDir+os.path.sep+"RMSTab.fits"
            if footprintLabel is not None:
                RMSTabFileName=RMSTabFileName.replace(".fits", "_%s.fits" % (footprintLabel))
            self.RMSTab=atpy.Table().read(RMSTabFileName)
            self.RMSDict={}
            tileNames=[]
            for tileName in self.tileNames:
                tileTab=self.RMSTab[self.RMSTab['tileName'] == tileName]
                if downsampleRMS == True and len(tileTab) > 0:
                    tileTab=downsampleRMSTab(tileTab) 
                if len(tileTab) > 0:    # We may have some blank tiles...
                    self.RMSDict[tileName]=tileTab
                    tileNames.append(tileName)
            self.tileNames=tileNames
            self.totalAreaDeg2=self.RMSTab['areaDeg2'].sum()
            
            # For weighting - arrays where entries correspond with tileNames list
            tileAreas=[]    
            for tileName in self.tileNames:
                areaDeg2=self.RMSTab[self.RMSTab['tileName'] == tileName]['areaDeg2'].sum()
                tileAreas.append(areaDeg2)
            self.tileAreas=np.array(tileAreas)
            self.fracArea=self.tileAreas/self.totalAreaDeg2

            # For quick sample generation
            self.mockOversampleFactor=mockOversampleFactor
            self.y0NoiseAverageDict={}
            for tileName in self.tileNames:
                RMSTab=self.RMSDict[tileName]
                areaWeights=RMSTab['areaDeg2'].data/RMSTab['areaDeg2'].data.sum()
                if areaWeights.sum() > 0:
                    self.y0NoiseAverageDict[tileName]=np.average(RMSTab['y0RMS'].data, 
                                                                 weights = areaWeights)            
            assert(len(self.tileNames) == len(self.tileAreas))
            assert(len(self.tileNames) == len(self.y0NoiseAverageDict.keys()))

            # For relativistic corrections
            self.fRelDict=signals.loadFRelWeights(self.selFnDir+os.path.sep+"fRelWeights.fits")
            
            # Initial cosmology set-up
            minMass=5e13
            zMin=0.0
            H0=self.scalingRelationDict['H0']
            Om0=self.scalingRelationDict['Om0']
            Ob0=self.scalingRelationDict['Ob0']
            sigma8=self.scalingRelationDict['sigma8']
            ns=self.scalingRelationDict['ns']
            self.mockSurvey=MockSurvey.MockSurvey(minMass, self.totalAreaDeg2, zMin, zMax, H0, Om0, Ob0, sigma8, ns,
                                                  zStep = self.zStep, enableDrawSample = enableDrawSample,
                                                  delta = delta, rhoType = rhoType, massFunction = massFunction)
            self.update(H0, Om0, Ob0, sigma8, ns)


    def _setUpAreaMask(self):
        """Sets-up WCS info and loads area masks - needed for quick position checks etc.
        
        """
        
        # This takes ~20 sec to set-up - we could cache, or it's overhead when initialising SelFn
        # We could do a lot of this by just making the area mask, mass limit maps etc. MEFs
        # But then we wouldn't want to lazily load them (probably)
        self.tileTab=atpy.Table()
        self.tileTab.add_column(atpy.Column(list(self.tileNames), 'tileName'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileNames)), 'RAMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileNames)), 'RAMax'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileNames)), 'decMin'))
        self.tileTab.add_column(atpy.Column(np.zeros(len(self.tileNames)), 'decMax'))
        self.WCSDict={}
        self.areaMaskDict={}
        for row in self.tileTab:
            areaMap, wcs=loadAreaMask(row['tileName'], self.selFnDir)
            self.WCSDict[row['tileName']]=wcs.copy()
            self.areaMaskDict[row['tileName']]=areaMap
            ra0, dec0=self.WCSDict[row['tileName']].pix2wcs(0, 0)
            ra1, dec1=self.WCSDict[row['tileName']].pix2wcs(wcs.header['NAXIS1'], wcs.header['NAXIS2'])
            if ra1 > ra0:
                ra1=-(360-ra1)
            row['RAMin']=min([ra0, ra1])
            row['RAMax']=max([ra0, ra1])
            row['decMin']=min([dec0, dec1])
            row['decMax']=max([dec0, dec1])
            
            
    def checkCoordsInAreaMask(self, RADeg, decDeg):
        """Checks if the given RA, dec coords are in valid regions of the map.
        
        Args:
            RADeg (:obj:`float` or :obj:`np.ndarray`): RA in decimal degrees.
            decDeg (:obj:`float` or :obj:`np.ndarray`): Dec in decimal degrees.
        
        Returns:
            `True` if the coordinates are in the area mask mask, `False` if not.
            
        """

        if self.tileTab is None:
            self._setUpAreaMask()

        RADeg=np.array(RADeg)
        decDeg=np.array(decDeg)
        if RADeg.shape == ():
            RADeg=[RADeg]
        if decDeg.shape == ():
            decDeg=[decDeg]
        inMaskList=[]
        for ra, dec in zip(RADeg, decDeg):
            inMask=False
            # Inside footprint check
            # NOTE: Tiles may have -ve RAMin coords
            raMask=np.logical_and(np.greater_equal(ra, self.tileTab['RAMin']), np.less(ra, self.tileTab['RAMax']))
            negRAMask=np.logical_and(np.greater_equal(-(360-ra), self.tileTab['RAMin']), np.less(-(360-ra), self.tileTab['RAMax']))
            raMask=np.logical_or(raMask, negRAMask)
            decMask=np.logical_and(np.greater_equal(dec, self.tileTab['decMin']), np.less(dec, self.tileTab['decMax']))
            tileMask=np.logical_and(raMask, decMask)
            # This is just dealing with bytes versus strings in python3
            matchTilesList=[]
            for item in self.tileTab['tileName'][tileMask].tolist():
                if type(item) == bytes:
                    matchTilesList.append(item.decode('utf-8'))
                else:
                    matchTilesList.append(str(item))
            for tileName in matchTilesList:
                x, y=self.WCSDict[tileName].wcs2pix(ra, dec)
                x=int(round(x)); y=int(round(y))
                if x < self.WCSDict[tileName].header['NAXIS1'] and y < self.WCSDict[tileName].header['NAXIS2'] \
                    and self.areaMaskDict[tileName][y, x] > 0:
                    inMask=True
            inMaskList.append(inMask)
        
        if len(inMaskList) > 1:
            return np.array(inMaskList)
        else:
            return inMaskList[0]
        

    def update(self, H0, Om0, Ob0, sigma8, ns, scalingRelationDict = None):
        """Re-calculates the survey-average selection function for a given set of cosmological and scaling
        relation parameters.
        
        Returns:
            None - attributes such as :attr:`compMz`, the (log\ :sub:`10` mass, z) completeness grid, are
            updated in-place.

        """

        if scalingRelationDict is not None:
            self.scalingRelationDict=scalingRelationDict
        
        self.mockSurvey.update(H0, Om0, Ob0, sigma8, ns)
        
        if self.method == 'injection':
            print("completeness from injection sim")
            import IPython
            IPython.embed()
            sys.exit()
            theta500s=np.unique(self.injTab['theta500Arcmin'])
            for i in range(len(theta500s)):
                mask=self.injTab['theta500Arcmin'] == theta500s[i]
                plt.hist(self.injTab['SNR'][mask], bins = 100, alpha = 0.5, label = theta500s[i])
                print(np.median(self.injTab['SNR'][mask]), np.median(self.injTab['inFlux'][mask]), np.median(self.injTab['outFlux'][mask]))
            plt.legend()

            # This is what the input distribution was
            amplitudeRange=[0.001, 1]
            amp=np.random.uniform(amplitudeRange[0], amplitudeRange[1], len(self.injTab))
            binEdges=np.linspace(amplitudeRange[0], amplitudeRange[1], 101)
            plt.hist(amp, bins = binEdges, density = True, alpha = 0.5)
            theta500s=np.unique(self.injTab['theta500Arcmin'])

            # Using KS test to find where distribution is consistent with uniform
            #for i in range(len(theta500s)):
            i=5
            print(theta500s[i])
            mask=self.injTab['theta500Arcmin'] == theta500s[i]
            #plt.hist(self.injTab['inFlux'][mask], bins = binEdges, density = True, alpha = 0.5, label = theta500s[i])
            #plt.legend()
            N, binEdges=np.histogram(self.injTab['inFlux'][mask], bins = binEdges, density = True)
            binCentres=(binEdges[1:]+binEdges[:-1])/2
            for b in binEdges:
                injInFlux=self.injTab['inFlux'][mask]
                a=stats.ks_2samp(injInFlux[injInFlux > b], amp[amp > b])
                if a.pvalue > 0.1:
                    break
            plt.hist(self.injTab['inFlux'][mask], bins = binEdges, density = True, alpha = 0.5, label = theta500s[i])
            plt.plot([b]*3, np.linspace(0, 1.5, 3), 'k--')
            plt.legend()


            #---
            # WARNING: Here there is no Q, y0 is true y0, not y0~
            zRange=self.mockSurvey.z
            y0GridCube=[]
            compMzCube=[]
            SNRCube=[]
            #theta500Cube=[]
            for tileName in self.RMSDict.keys():
                tileName='1_10_8'
                tenToA0, B0, Mpivot, sigma_int=[self.scalingRelationDict['tenToA0'], self.scalingRelationDict['B0'],
                                                self.scalingRelationDict['Mpivot'], self.scalingRelationDict['sigma_int']]
                y0Grid=np.zeros([zRange.shape[0], self.mockSurvey.clusterCount.shape[1]])
                SNRGrid=np.zeros([zRange.shape[0], self.mockSurvey.clusterCount.shape[1]])
                #theta500Grid=np.zeros([zRange.shape[0], self.mockSurvey.clusterCount.shape[1]])
                self.RMSDict[tileName]
                RMSTab=self.RMSDict[tileName]
                areaWeights=RMSTab['areaDeg2']/RMSTab['areaDeg2'].sum()
                tileRMSValue=np.average(RMSTab['y0RMS'], weights = areaWeights)
                for i in range(len(zRange)):
                    zk=zRange[i]
                    k=np.argmin(abs(self.mockSurvey.z-zk))
                    if self.mockSurvey.delta != 500 or self.mockSurvey.rhoType != "critical":
                        log10M500s=np.log10(self.mockSurvey.mdef.translate_mass(self.mockSurvey.cosmoModel,
                                                                                self.mockSurvey.M,
                                                                                self.mockSurvey.a[k],
                                                                                self.mockSurvey._M500cDef))
                    else:
                        log10M500s=self.mockSurvey.log10M
                    #theta500s_zk=interpolate.splev(log10M500s, self.mockSurvey.theta500Splines[k])
                    #Qs_zk=self.Q.getQ(theta500s_zk, zk, tileName = tileName)
                    true_y0s_zk=tenToA0*np.power(self.mockSurvey.Ez[k], 2)*np.power(np.power(10, self.mockSurvey.log10M)/Mpivot,
                                                                                    1+B0)#*Qs_zk
                    if self.applyRelativisticCorrection == True:
                        fRels_zk=interpolate.splev(log10M500s, self.mockSurvey.fRelSplines[k])
                        true_y0s_zk=true_y0s_zk*fRels_zk
                    y0Grid[i]=true_y0s_zk #*0.95
                SNRGrid=y0Grid/tileRMSValue
                # WARNING: true y0 SNR (NOT y0~ SNR)
                self.injTab['in_SNR']=self.injTab['inFlux']/self.injTab['noiseLevel']


                #areaWeightedRMS=
                for i in range(len(RMSTab)):
                    RMSTab['y0RMS'][i]

                    #---
                    # No intrinsic scatter, Gaussian noise
                    sfi=stats.norm.sf(y0Lim[i], loc = y0Grid, scale = RMSTab['y0RMS'][i])
                    compMz=compMz+sfi*areaWeights[i]
                    #---
                    # Old
                    #SNRGrid=y0Grid/RMSTab['y0RMS'][i]
                    #log_y0Err=1/SNRGrid
                    #log_y0Err[SNRGrid < self.SNRCut]=1/self.SNRCut
                    #log_totalErr=np.sqrt(log_y0Err**2 + sigma_int**2)
                    #sfi=stats.norm.sf(log_y0Lim[i], loc = log_y0, scale = log_totalErr)
                    #compMz=compMz+sfi*areaWeights[i]
                if self.maxTheta500Arcmin is not None:
                    compMz=compMz*np.array(theta500Grid < self.maxTheta500Arcmin, dtype = float)

                    SNRGrid[i]=y0Grid[i]/RMSTab['y0RMS'][i]
                    #theta500Grid[i]=theta500s_zk

            #---
            theta500s=np.unique(self.injTab['theta500Arcmin'])
            for i in range(len(theta500s)):
                mask=self.injTab['theta500Arcmin'] == theta500s[i]
                plt.plot(self.injTab['SNR'][mask], self.injTab['outFlux'][mask], '.', label = theta500s[i])


            #---
            # Playing around to start
            # Q from the source injection sims
            theta500s=np.unique(self.injTab['theta500Arcmin'])
            Q=np.zeros(len(theta500s))
            QHigh=np.zeros(len(theta500s))
            QLow=np.zeros(len(theta500s))
            for i in range(len(theta500s)):
                mask=self.injTab['theta500Arcmin'] == theta500s[i]
                Q[i]=np.percentile(self.injTab['outFlux'][mask]/self.injTab['inFlux'][mask], 50)
                QHigh[i]=np.percentile(self.injTab['outFlux'][mask]/self.injTab['inFlux'][mask], 90)
                QLow[i]=np.percentile(self.injTab['outFlux'][mask]/self.injTab['inFlux'][mask], 10)

            # Q comparison
            #plotSettings.update_rcParams()
            #plt.figure(figsize = (10, 8))
            #plt.errorbar(theta500s, Q, yerr = [Q-QLow, QHigh-Q], fmt = 'ko', label = 'all tiles source inj')
            #plt.plot(theta500s, self.Q.getQ(theta500s, tileName = '1_10_8'), label = 'QFit: 1_10_8')
            #plt.legend()
            #plt.xlabel('$\\theta_{\\rm 500c}$ (arcmin)')
            #plt.ylabel("$Q$")

            # Interpolator
            injQtck=interpolate.splrep(theta500s, Q)

            #---

        elif self.method == 'fast':
            zRange=self.mockSurvey.z
            y0GridCube=[]
            compMzCube=[]
            theta500Cube=[]
            for tileName in self.RMSDict.keys():
                tenToA0, B0, Mpivot, sigma_int=[self.scalingRelationDict['tenToA0'], self.scalingRelationDict['B0'],
                                                self.scalingRelationDict['Mpivot'], self.scalingRelationDict['sigma_int']]
                y0Grid=np.zeros([zRange.shape[0], self.mockSurvey.clusterCount.shape[1]])
                theta500Grid=np.zeros([zRange.shape[0], self.mockSurvey.clusterCount.shape[1]])
                for i in range(len(zRange)):
                    zk=zRange[i]
                    k=np.argmin(abs(self.mockSurvey.z-zk))
                    if self.mockSurvey.delta != 500 or self.mockSurvey.rhoType != "critical":
                        log10M500s=np.log10(self.mockSurvey.mdef.translate_mass(self.mockSurvey.cosmoModel,
                                                                                self.mockSurvey.M,
                                                                                self.mockSurvey.a[k],
                                                                                self.mockSurvey._M500cDef))
                    else:
                        log10M500s=self.mockSurvey.log10M
                    theta500s_zk=interpolate.splev(log10M500s, self.mockSurvey.theta500Splines[k])
                    Qs_zk=self.Q.getQ(theta500s_zk, zk, tileName = tileName)
                    true_y0s_zk=tenToA0*np.power(self.mockSurvey.Ez[k], 2)*np.power(np.power(10, self.mockSurvey.log10M)/Mpivot,
                                                                                    1+B0)*Qs_zk
                    if self.applyRelativisticCorrection == True:
                        fRels_zk=interpolate.splev(log10M500s, self.mockSurvey.fRelSplines[k])
                        true_y0s_zk=true_y0s_zk*fRels_zk
                    y0Grid[i]=true_y0s_zk #*0.95
                    theta500Grid[i]=theta500s_zk

                # For some cosmological parameters, we can still get the odd -ve y0
                y0Grid[y0Grid <= 0] = 1e-9

                # Calculate completeness using area-weighted average
                # NOTE: RMSTab that is fed in here can be downsampled in noise resolution for speed
                RMSTab=self.RMSDict[tileName]
                areaWeights=RMSTab['areaDeg2']/RMSTab['areaDeg2'].sum()
                y0Lim=self.SNRCut*RMSTab['y0RMS']
                log_y0Lim=np.log(self.SNRCut*RMSTab['y0RMS'])
                log_y0=np.log(y0Grid)
                compMz=np.zeros(log_y0.shape)
                for i in range(len(RMSTab)):
                    #---
                    # No intrinsic scatter, Gaussian noise
                    sfi=stats.norm.sf(y0Lim[i], loc = y0Grid, scale = RMSTab['y0RMS'][i])
                    compMz=compMz+sfi*areaWeights[i]
                    #---
                    # Old
                    #SNRGrid=y0Grid/RMSTab['y0RMS'][i]
                    #log_y0Err=1/SNRGrid
                    #log_y0Err[SNRGrid < self.SNRCut]=1/self.SNRCut
                    #log_totalErr=np.sqrt(log_y0Err**2 + sigma_int**2)
                    #sfi=stats.norm.sf(log_y0Lim[i], loc = log_y0, scale = log_totalErr)
                    #compMz=compMz+sfi*areaWeights[i]
                if self.maxTheta500Arcmin is not None:
                    compMz=compMz*np.array(theta500Grid < self.maxTheta500Arcmin, dtype = float)
                compMzCube.append(compMz)
                y0GridCube.append(y0Grid)
            compMzCube=np.array(compMzCube)
            y0GridCube=np.array(y0GridCube)
            self.compMz=np.average(compMzCube, axis = 0, weights = self.fracArea)
            self.y0Grid=np.average(y0GridCube, axis = 0, weights = self.fracArea)

            #print("fix in selFn completeness")
            #import IPython
            #IPython.embed()
            #sys.exit()

            ##---
            ## Old
            #compMzCube=[]
            #count=0
            #method='montecarlo'
            #for tileName in self.RMSDict.keys():
                #count=count+1
                #if method == 'montecarlo':
                    #print("... %d/%d ..." % (count, len(self.tileNames)))
                    #compMzCube.append(calcCompleteness(self.RMSDict[tileName], self.SNRCut, tileName,
                                                    #self.mockSurvey, self.scalingRelationDict, self.Q,
                                                    #method = 'montecarlo', numDraws = 100000, numIterations = 5))
                #else:
                    #compMzCube.append(calcCompleteness(self.RMSDict[tileName], self.SNRCut, tileName,
                                                    #self.mockSurvey, self.scalingRelationDict, self.Q))

                #if np.any(np.isnan(compMzCube[-1])) == True:
                    #raise Exception("NaNs in compMz for tile '%s'" % (tileName))
            #compMzCube=np.array(compMzCube)
            #self.compMz=np.average(compMzCube, axis = 0, weights = self.fracArea)
        

    def projectCatalogToMz(self, tab):
        """Project a Nemo cluster catalog (an astropy Table) into the (log\ :sub:`10` mass, z) grid, taking
        into account the uncertainties on y0, and redshift. Note that if the redshift error is non-zero, this
        is a lot slower.
        
        Args:
            tab (:obj:`astropy.table.Table`): A Nemo cluster catalog, containing `redshift` and `redshiftErr`
                columns.
                
        Returns:
            A 2d array containing the projection of the catalog on the (log\ :sub:`10` mass, z) grid.
        
        """
        
        catProjectedMz=np.zeros(self.mockSurvey.clusterCount.shape)
        tenToA0, B0, Mpivot, sigma_int=self.scalingRelationDict['tenToA0'], self.scalingRelationDict['B0'], \
                                       self.scalingRelationDict['Mpivot'], self.scalingRelationDict['sigma_int']

        for row in tab:
            tileName=row['tileName']
            z=row['redshift']
            zErr=row['redshiftErr']
            y0=row['fixed_y_c']*1e-4
            y0Err=row['fixed_err_y_c']*1e-4
            P=signals.calcPMass(y0, y0Err, z, zErr, self.Q, self.mockSurvey, 
                                  tenToA0 = tenToA0, B0 = B0, Mpivot = Mpivot, sigma_int = sigma_int, 
                                  applyMFDebiasCorrection = self.applyMFDebiasCorrection, 
                                  fRelWeightsDict = self.fRelDict[tileName],
                                  return2D = True, tileName = tileName)
            # Paste into (M, z) grid
            catProjectedMz=catProjectedMz+P # For return2D = True, P is normalised such that 2D array sum is 1
        
        return catProjectedMz


    def projectCatalogToMz_simple(self, tab):
        """Project a Nemo cluster catalog (an astropy Table) into the (log\ :sub:`10` mass, z) grid. This version
        doesn't take into account any uncertainties (which may be okay if your binning is coarse enough).
                
        Args:
            tab (:obj:`astropy.table.Table`): A Nemo cluster catalog, containing `redshift` and `redshiftErr`
                columns.
                
        Returns:
            A 2d array containing the projection of the catalog on the (log\ :sub:`10` mass, z) grid.
        
        """
        
        tenToA0, B0, Mpivot, sigma_int=self.scalingRelationDict['tenToA0'], self.scalingRelationDict['B0'], \
                                       self.scalingRelationDict['Mpivot'], self.scalingRelationDict['sigma_int']
                                           
        obs_log10Ms=[]
        for row in tab:
            tileName=row['tileName']
            z=row['redshift']
            zErr=row['redshiftErr']
            y0=row['fixed_y_c']*1e-4
            y0Err=row['fixed_err_y_c']*1e-4
            massDict=signals.calcMass(y0, y0Err, z, zErr, self.Q, tenToA0 = tenToA0, B0 = B0, Mpivot = Mpivot, 
                                            sigma_int = sigma_int, mockSurvey = self.mockSurvey, 
                                            applyMFDebiasCorrection = self.applyMFDebiasCorrection,
                                            fRelWeightsDict = self.fRelDict[tileName],
                                            calcErrors = False, tileName = tileName)
            obs_log10Ms.append(14+np.log10(massDict['M500']))
        obsGrid, obs_log10MBinEdges, obs_zBinEdges=np.histogram2d(obs_log10Ms, tab['redshift'], 
                                                                  bins = [self.mockSurvey.log10MBinEdges, 
                                                                          self.mockSurvey.zBinEdges])
        obsGrid=obsGrid.transpose()
            
        return obsGrid


    def addPDetToCatalog(self, tab):
        """Given a catalog, add a column named `Pdet`, containing the detection probability.
        
        Args:
            tab (:obj:`astropy.table.Table`): A Nemo cluster catalog.
                
        Returns:
            Catalog with `Pdet` column added (:obj:`astropy.table.Table`)
        
        """

        log_y0Lim=np.log(self.SNRCut*tab['fixed_err_y_c']*1e-4)
        log_y0=np.log(tab['fixed_y_c']*1e-4)
        log_y0Err=1/tab['fixed_SNR']
        sigma_int=self.scalingRelationDict['sigma_int']
        log_totalErr=np.sqrt(log_y0Err**2 + sigma_int**2)
        Pdet=np.zeros(len(tab))
        for i in range(len(Pdet)):
            Pdet[i]=stats.norm.sf(log_y0Lim[i], loc = log_y0[i], scale = log_totalErr[i])
        tab['Pdet']=Pdet
        
        return tab

    
    def generateMockSample(self, mockOversampleFactor = None, applyPoissonScatter = True):
        """Returns a mock catalog (but with no object coordinate information).
        
        Args:
            mockOversampleFactor (:obj:`float`, optional): Scale the number of objects in the mock by this
                factor (e.g., this can be used to apply an overall area re-scaling). If given, this
                overrides self.mockOversampleFactor.
            applyPoissonScatter (:obj:`bool`, optional): If True, apply Poisson scatter to the cluster
                number counts when generating the mock catalog.

        Note:
            This currently uses the average noise level in each tile, rather than the full noise
            distribution in each tile.
        
        """
        
        if mockOversampleFactor is None:
            mockOversampleFactor=self.mockOversampleFactor

        mockTabsList=[]
        for tileName, areaDeg2 in zip(self.tileNames, self.tileAreas):
            mockTab=self.mockSurvey.drawSample(self.y0NoiseAverageDict[tileName], self.scalingRelationDict, 
                                               self.Q, wcs = None, 
                                               photFilterLabel = self.photFilterLabel, tileName = tileName, 
                                               makeNames = False,
                                               SNRLimit = self.SNRCut, applySNRCut = True,
                                               areaDeg2 = areaDeg2*mockOversampleFactor,
                                               applyPoissonScatter = applyPoissonScatter,
                                               applyIntrinsicScatter = True,
                                               applyNoiseScatter = True,
                                               applyRelativisticCorrection = self.applyRelativisticCorrection)
            if mockTab is not None and len(mockTab) > 0:
                mockTabsList.append(mockTab)
        tab=atpy.vstack(mockTabsList)

        return tab
    
    
    def getMassLimit(self, completenessFraction, zBinEdges = None):
        """Return the mass limit (units of 10\ :sup:`14` MSun) as a function of redshift, for the given
        completeness level. 
        
        Args:
            completenessFraction (:obj:`float`): The completeness fraction (a number between 0 and 1) at
                which to return the mass limit.
            zBinEdges (:obj:`np.ndarray`, optional): The redshifts at which the completeness will be
                evaluated. If not given, :attr:`self.mockSurvey.z` will be used.
                
        Returns:
            Mass completeness by redshift (:obj:`np.ndarray`).
        
        """
        
        if zBinEdges is None:
            zBinEdges=self.mockSurvey.zBinEdges
            
        return calcMassLimit(completenessFraction, self.compMz, self.mockSurvey, zBinEdges)

    
#------------------------------------------------------------------------------------------------------------
def loadAreaMask(tileName, selFnDir):
    """Loads the survey area mask, i.e., the area searched for sources and clusters, for the given tile.
    
    Args:
        tileName (:obj:`str`): The name of the tile for which the area mask will be loaded.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
    
    Returns:
        Map array (2d :obj:`np.ndarray`), WCS object (:obj:`astWCS.WCS`)
    
    """
    
    areaMap, wcs=_loadTile(tileName, selFnDir, "areaMask", extension = 'fits')   
    return areaMap, wcs

#------------------------------------------------------------------------------------------------------------
def loadFlagMask(tileName, selFnDir):
    """Loads the flag mask, i.e., areas flagged for reasons such as point source subtraction, for the given
    tile.

    Args:
        tileName (:obj:`str`): The name of the tile for which the flag mask will be loaded.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.

    Returns:
        Map array (2d :obj:`np.ndarray`), WCS object (:obj:`astWCS.WCS`)

    """

    areaMap, wcs=_loadTile(tileName, selFnDir, "flagMask", extension = 'fits')
    return areaMap, wcs

#------------------------------------------------------------------------------------------------------------
def loadRMSMap(tileName, selFnDir, photFilter):
    """Loads the RMS (noise) map for the given tile.
    
    Args:
        tileName (:obj:`str`): The name of the tile for which the RMS (noise) map will be loaded.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        photFilter (:obj:`str`): Name of the reference filter, as specified in, e.g., a :ref:`nemoCommand`
            config file (see :ref:`ConfigReference`).

    Returns:
        Map array (2d :obj:`np.ndarray`), WCS object (:obj:`astWCS.WCS`)
    
    """
    
    RMSMap, wcs=_loadTile(tileName, selFnDir, "RMSMap_%s" % (photFilter), extension = 'fits')   
    return RMSMap, wcs

#------------------------------------------------------------------------------------------------------------
def loadMassLimitMap(tileName, diagnosticsDir, z):
    """Loads the mass limit map for the given tile at the given redshift.

    Args:
        tileName (:obj:`str`): The name of the tile for which the mass limit map will be loaded.
        diagnosticsDir (:obj:`str`): Path to the ``diagnostics/`` directory, as produced by the 
            :ref:`nemoCommand` command.
        z (:obj:`float`): Redshift at which the mass limit map was made (should match an entry in the
            :ref:`nemoCommand` config file).

    Returns:
        Map array (2d :obj:`np.ndarray`), WCS object (:obj:`astWCS.WCS`)
    
    """
    
    massLimMap, wcs=_loadTile(tileName, diagnosticsDir, "massLimitMap_z%s" % (str(z).replace(".", "p")), 
                              extension = 'fits')   
    return massLimMap, wcs

#------------------------------------------------------------------------------------------------------------
def loadIntersectionMask(tileName, selFnDir, footprint):
    """Loads the intersection mask for the given tile and footprint.
    
    Args:
        tileName (:obj:`str`): The name of the tile for which the intersection mask will be loaded.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        footprint (:obj:`str`): The name of the footprint for which the intersection will be calculated,
            as defined in the :ref:`nemoCommand` config file (see :ref:`selFnFootprints`).
    
    Returns:
        Map array (2d :obj:`np.ndarray`), WCS object (:obj:`astWCS.WCS`)
    
    """           
            
    intersectMask, wcs=_loadTile(tileName, selFnDir, "intersect_%s" % (footprint), extension = 'fits')   
    return intersectMask, wcs

#------------------------------------------------------------------------------------------------------------
def _loadTile(tileName, baseDir, baseFileName, extension = 'fits'):
    """Generic function to load a tile image from either a multi-extension FITS file, or a file with 
    #tileName.fits type extension, whichever is found.
        
    Returns map array, wcs
    
    """
    
    # After tidyUp is run, this will be a MEF file... during first run, it won't be (written under MPI)
    if os.path.exists(baseDir+os.path.sep+"%s#%s.%s" % (baseFileName, tileName, extension)):
        fileName=baseDir+os.path.sep+"%s#%s.%s" % (baseFileName, tileName, extension)
    elif os.path.exists(baseDir+os.path.sep+tileName+os.path.sep+"%s#%s.%s" % (baseFileName, tileName, extension)):
        # Lustre-friendlier: each tile has its own directory
        fileName=baseDir+os.path.sep+tileName+os.path.sep+"%s#%s.%s" % (baseFileName, tileName, extension)
    else:
        fileName=baseDir+os.path.sep+"%s.%s" % (baseFileName, extension)
    with pyfits.open(fileName) as img:
        # If we find the tile - great. If not, we use first extension with data as it'll be compressed
        if tileName in img:
            extName=tileName
            data=img[extName].data
        else:
            data=None
        if data is None:
            for extName in img:
                data=img[extName].data
                if data is not None:
                    break
        data=img[extName].data
        wcs=astWCS.WCS(img[extName].header, mode = 'pyfits')
    
    return data, wcs

#------------------------------------------------------------------------------------------------------------
def getTileTotalAreaDeg2(tileName, selFnDir, masksList = [], footprintLabel = None):
    """Returns the total area of the tile given by `tileName` (taking into account masked regions).
    
    Args:
        tileName (:obj:`str`): The name of the tile for which the area will be calculated.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        masksList (:obj:`list`, optional): A list of paths to FITS-format mask images, which contain pixels
            with values of 1 to indicate valid survey area, 0 otherwise. If given, the area is calculated
            for the intersection of these masks with the survey area mask.
        footprintLabel (:obj:`str`, optional): The name of the footprint for which the intersection will
            be calculated, as defined in the :ref:`nemoCommand` config file (see :ref:`selFnFootprints`).
    
    Returns:
        Tile area, after masking, in square degrees.
    
    """
    
    areaMap, wcs=loadAreaMask(tileName, selFnDir)
    areaMapSqDeg=(maps.getPixelAreaArcmin2Map(areaMap.shape, wcs)*areaMap)/(60**2)
    totalAreaDeg2=areaMapSqDeg.sum()
    
    if footprintLabel is not None:  
        intersectMask=makeIntersectionMask(tileName, selFnDir, footprintLabel, masksList = masksList)
        totalAreaDeg2=(areaMapSqDeg*intersectMask).sum()        
        
    return totalAreaDeg2

#------------------------------------------------------------------------------------------------------------
def makeIntersectionMask(tileName, selFnDir, label, masksList = []):
    """Creates an intersection mask between the survey mask, and the mask files given in `masksList`.
    
    Args:
        tileName (:obj:`str`): The name of the tile for which the intersection mask will be made (or loaded
            from disk, if cached).
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        label (:obj:`str`): The name of the footprint for which the intersection will be calculated,
            as defined in the :ref:`nemoCommand` config file (see :ref:`selFnFootprints`).
        masksList (:obj:`list`, optional): A list of paths to FITS-format mask images, which contain pixels
            with values of 1 to indicate valid survey area, 0 otherwise. If given, the area is calculated
            for the intersection of these masks with the survey area mask.
    
    Returns:
        Intersection mask (1 = valid area, 0 = outside of intersection area; 2d :obj:`np.ndarray`)
        
    Note:
        For speed, it is assumed that the declination axis is aligned with the vertical axis in each
        mask image. This routine caches the intersection masks in `selFnDir`.
            
    """
    
    # After tidyUp has run, there will be intersection mask MEF files
    intersectFileName=selFnDir+os.path.sep+"intersect_%s.fits" % (label)
    if os.path.exists(intersectFileName):
        intersectMask, wcs=loadIntersectionMask(tileName, selFnDir, label)
        return intersectMask
    
    # Otherwise, we may have a per-tile intersection mask
    intersectFileName=selFnDir+os.path.sep+"intersect_%s#%s.fits" % (label, tileName)
    if os.path.exists(intersectFileName):
        intersectMask, wcs=loadIntersectionMask(tileName, selFnDir, label)
        return intersectMask

    # Otherwise... make it
    areaMap, wcs=loadAreaMask(tileName, selFnDir)
    RAMin, RAMax, decMin, decMax=wcs.getImageMinMaxWCSCoords()
    if masksList == []:
        raise Exception("didn't find previously cached intersection mask but makeIntersectionMask called with empty masksList")
    print("... creating %s intersection mask (%s) ..." % (label, tileName))         
    intersectMask=np.zeros(areaMap.shape)
    outRACoords=np.array(wcs.pix2wcs(np.arange(intersectMask.shape[1]), [0]*intersectMask.shape[1]))
    outDecCoords=np.array(wcs.pix2wcs([0]*np.arange(intersectMask.shape[0]), np.arange(intersectMask.shape[0])))
    outRA=outRACoords[:, 0]
    outDec=outDecCoords[:, 1]
    RAToX=interpolate.interp1d(outRA, np.arange(intersectMask.shape[1]), fill_value = 'extrapolate')
    DecToY=interpolate.interp1d(outDec, np.arange(intersectMask.shape[0]), fill_value = 'extrapolate')
    for fileName in masksList:
        with pyfits.open(fileName) as maskImg:
            for hdu in maskImg:
                if type(hdu) == pyfits.ImageHDU:
                    break
            maskWCS=astWCS.WCS(hdu.header, mode = 'pyfits')
            maskData=hdu.data
        # From sourcery tileDir stuff
        RAc, decc=wcs.getCentreWCSCoords()
        xc, yc=maskWCS.wcs2pix(RAc, decc)
        xc, yc=int(xc), int(yc)
        xIn=np.arange(maskData.shape[1])
        yIn=np.arange(maskData.shape[0])
        inRACoords=np.array(maskWCS.pix2wcs(xIn, [yc]*len(xIn)))
        inDecCoords=np.array(maskWCS.pix2wcs([xc]*len(yIn), yIn))
        inRA=inRACoords[:, 0]
        inDec=inDecCoords[:, 1]
        RAToX=interpolate.interp1d(inRA, xIn, fill_value = 'extrapolate')
        DecToY=interpolate.interp1d(inDec, yIn, fill_value = 'extrapolate')
        outRACoords=np.array(wcs.pix2wcs(np.arange(intersectMask.shape[1]), [0]*intersectMask.shape[1]))
        outDecCoords=np.array(wcs.pix2wcs([0]*np.arange(intersectMask.shape[0]), np.arange(intersectMask.shape[0])))
        outRA=outRACoords[:, 0]
        outDec=outDecCoords[:, 1]
        xIn=np.array(RAToX(outRA), dtype = int)
        yIn=np.array(DecToY(outDec), dtype = int)
        xMask=np.logical_and(xIn >= 0, xIn < maskData.shape[1])
        yMask=np.logical_and(yIn >= 0, yIn < maskData.shape[0])
        xOut=np.arange(intersectMask.shape[1])
        yOut=np.arange(intersectMask.shape[0])
        for i in yOut[yMask]:
            intersectMask[i][xMask]=maskData[yIn[i], xIn[xMask]]
    intersectMask=np.array(np.greater(intersectMask, 0.5), dtype = int)
    maps.saveFITS(intersectFileName, intersectMask, wcs, compressionType = 'PLIO_1')
        
    return intersectMask

#------------------------------------------------------------------------------------------------------------
def getRMSTab(tileName, photFilterLabel, selFnDir, footprintLabel = None):
    """Makes a table containing map area in the tile refered to by `tileName` against RMS (noise level)
    values, compressing the information in the RMS maps. Results are cached under `selFnDir`, and read from
    disk if found.
    
    Args:
        tileName (:obj:`str`): The name of the tile.
        photFilterLabel (:obj:`str`): Name of the reference filter, as specified in, e.g., a 
            :ref:`nemoCommand` config file (see :ref:`ConfigReference`).
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        footprintLabel (:obj:`str`, optional): The name of the footprint in which the calculation will be
            done, as defined in the :ref:`nemoCommand` config file (see :ref:`selFnFootprints`).
    
    Returns:
        A table of RMS (noise level) values versus area in square degrees (:obj:`astropy.table.Table`).
    
    """
    
    # After tidyUp has run, we may have one global RMS table with an extra tileName column we can use
    RMSTabFileName=selFnDir+os.path.sep+"RMSTab.fits"
    if footprintLabel is not None:
        RMSTabFileName=RMSTabFileName.replace(".fits", "_%s.fits" % (footprintLabel))
    if os.path.exists(RMSTabFileName):
        tab=atpy.Table().read(RMSTabFileName)
        return tab[np.where(tab['tileName'] == tileName)]

    # Table doesn't exist, so make it...
    print(("... making RMS table for tile = %s, footprint = %s" % (tileName, footprintLabel)))
    RMSMap, wcs=loadRMSMap(tileName, selFnDir, photFilterLabel)
    areaMap, wcs=loadAreaMask(tileName, selFnDir)
    areaMapSqDeg=(maps.getPixelAreaArcmin2Map(areaMap.shape, wcs)*areaMap)/(60**2)

    if footprintLabel != None:  
        intersectMask=makeIntersectionMask(tileName, selFnDir, footprintLabel)
        areaMapSqDeg=areaMapSqDeg*intersectMask        
        RMSMap=RMSMap*intersectMask

    RMSValues=np.unique(RMSMap[np.nonzero(RMSMap)])

    totalAreaDeg2=areaMapSqDeg.sum()
    tileArea=np.zeros(len(RMSValues))
    for i in range(len(RMSValues)):
        tileArea[i]=areaMapSqDeg[np.equal(RMSMap, RMSValues[i])].sum()
    RMSTab=atpy.Table()
    RMSTab.add_column(atpy.Column(tileArea, 'areaDeg2'))
    RMSTab.add_column(atpy.Column(RMSValues, 'y0RMS'))
    # Checks - these should be impossible but we have seen (e.g., when messed up masks)
    tol=0.003
    if abs(RMSTab['areaDeg2'].sum()-areaMapSqDeg.sum()) > tol:
        raise Exception("Mismatch between area map and area in RMSTab for tile '%s'" % (tileName))
    if np.less(RMSTab['areaDeg2'], 0).sum() > 0:
        raise Exception("Negative area in tile '%s' - check your survey mask (and delete/remake tileDir files if necessary)." % (tileName))
    RMSTab.meta['NEMOVER']=nemo.__version__

    return RMSTab

#------------------------------------------------------------------------------------------------------------
def downsampleRMSTab(RMSTab, stepSize = 0.001*1e-4):
    """Downsamples `RMSTab` (see :meth:`getRMSTab`) in terms of noise resolution, binning by `stepSize`.
    
    Args:
        RMSTab (:obj:`astropy.table.Table`): An RMS table, as produced by :meth:`getRMSTab`.
        stepSize (:obj:`float`, optional): Sets the re-binning in terms of y0.
        
    Returns:
        A table of RMS (noise level) values versus area in square degrees (:obj:`astropy.table.Table`).        
    
    """
    
    binEdges=np.arange(RMSTab['y0RMS'].min(), RMSTab['y0RMS'].max()+stepSize, stepSize)
    y0Binned=[]
    tileAreaBinned=[]
    binMins=[]
    binMaxs=[]
    for i in range(len(binEdges)-1):
        mask=np.logical_and(RMSTab['y0RMS'] >= binEdges[i], RMSTab['y0RMS'] < binEdges[i+1])
        if mask.sum() > 0:
            y0Binned.append(np.average(RMSTab['y0RMS'][mask], weights = RMSTab['areaDeg2'][mask]))
            tileAreaBinned.append(np.sum(RMSTab['areaDeg2'][mask]))
            binMins.append(binEdges[i])
            binMaxs.append(binEdges[i+1])
    newRMSTab=atpy.Table()
    newRMSTab.add_column(atpy.Column(y0Binned, 'y0RMS'))
    newRMSTab.add_column(atpy.Column(tileAreaBinned, 'areaDeg2'))
    RMSTab=newRMSTab
        
    return RMSTab

#------------------------------------------------------------------------------------------------------------
def calcTileWeightedAverageNoise(tileName, photFilterLabel, selFnDir, footprintLabel = None):
    """Returns the area weighted average ỹ\ :sub:`0` noise value in the tile.
    
    Args:
        tileName (:obj:`str`): The name of the tile.
        photFilterLabel (:obj:`str`): Name of the reference filter, as specified in, e.g., a 
            :ref:`nemoCommand` config file (see :ref:`ConfigReference`).
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        footprintLabel (:obj:`str`, optional): The name of the footprint in which the calculation will be
            done, as defined in the :ref:`nemoCommand` config file (see :ref:`selFnFootprints`).
    
    Returns:
        Area-weighted average noise in ỹ\ :sub:`0` (``fixed_y_c`` in Nemo cluster catalogs).
        
    """

    RMSTab=getRMSTab(tileName, photFilterLabel, selFnDir, footprintLabel = footprintLabel)
    RMSValues=np.array(RMSTab['y0RMS'])
    tileArea=np.array(RMSTab['areaDeg2'])
    tileRMSValue=np.average(RMSValues, weights = tileArea)

    return tileRMSValue

#------------------------------------------------------------------------------------------------------------
def completenessByFootprint(selFnCollection, mockSurvey, diagnosticsDir, additionalLabel = ""):
    """Write out the average (log\ :sub:`10` mass, z) grid over all survey footprints provided in 
    `selFnCollection`, weighted by fraction of total survey area within the footprint. Also prints some
    useful statistics and produces some plots that are written to the `diagnosticsDir` directory.
    
    Args:
        selFnCollection (:obj:`dict`): A dictionary where each key points to a :class:`SelFn` object that
            describes a given survey footprint. Here, "full" corresponds to the whole survey, while "DES"
            may represent the DES footprint, if it has been defined in the :ref:`nemoCommand` config file
            (see :ref:`selFnFootprints`).
        mockSurvey (:class:`nemo.MockSurvey.MockSurvey`): A :class:`MockSurvey` object, used for halo mass
            function calculations and generating mock catalogs.
        diagnosticsDir (:obj:`str`): The path to the diagnostics directory, where the output from this
            routine will be written.
        additionalLabel (:obj:`str`, optional): This will be added to the output filenames (use this for
            e.g., tagging with the :meth:`calcCompleteness` method used).
    
    Returns:
        None
        
    Note:
        Output is written to files named e.g. ``diagnosticsDir/MzCompleteness_label.npz``, where `label`
        is the footprint name (i.e., a key in `selFnCollection`); 'full' is the default (survey-wide
        average).
    
    """
        
    zBinEdges=np.arange(0.05, 2.1, 0.1)
    zBinCentres=(zBinEdges[:-1]+zBinEdges[1:])/2.
    massLabel=mockSurvey.mdefLabel
        
    for footprintLabel in selFnCollection.keys():
        print(">>> Survey-averaged results inside footprint: %s ..." % (footprintLabel))
        selFnDictList=selFnCollection[footprintLabel]
        tileAreas=[]
        compMzCube=[]
        completeness=[]
        for selFnDict in selFnDictList:
            tileAreas.append(selFnDict['tileAreaDeg2'])
            massLimit_90Complete=calcMassLimit(0.9, selFnDict['compMz'], mockSurvey, zBinEdges = zBinEdges)
            completeness.append(massLimit_90Complete)
            compMzCube.append(selFnDict['compMz'])
        tileAreas=np.array(tileAreas)
        completeness=np.array(completeness)
        if np.sum(tileAreas) == 0:
            print("... no overlapping area with %s ..." % (footprintLabel))
            continue
        fracArea=tileAreas/np.sum(tileAreas)
        compMzCube=np.array(compMzCube)
        compMz_surveyAverage=np.average(compMzCube, axis = 0, weights = fracArea)

        outFileName=diagnosticsDir+os.path.sep+"MzCompleteness_%s%s.npz" % (footprintLabel, additionalLabel)
        np.savez(outFileName, z = mockSurvey.z, log10M500c = mockSurvey.log10M, 
                 M500Completeness = compMz_surveyAverage)
        
        makeMzCompletenessPlot(compMz_surveyAverage, mockSurvey.log10M, mockSurvey.z, footprintLabel, massLabel,
                               diagnosticsDir+os.path.sep+"MzCompleteness_%s%s.pdf" % (footprintLabel, additionalLabel))

        # 90% mass completeness limit and plots
        massLimit_90Complete=np.average(completeness, axis = 0, weights = fracArea)  # agrees with full mass limit map
        makeMassLimitVRedshiftPlot(massLimit_90Complete, zBinCentres, diagnosticsDir+os.path.sep+"completeness90Percent_%s%s.pdf" % (footprintLabel, additionalLabel), title = "footprint: %s" % (footprintLabel))
        zMask=np.logical_and(zBinCentres >= 0.2, zBinCentres < 1.0)
        averageMassLimit_90Complete=np.average(massLimit_90Complete[zMask])
        print("... total survey area (after masking) = %.1f sq deg" % (np.sum(tileAreas)))
        print("... survey-averaged 90%% mass completeness limit (z = 0.5) = %.1f x 10^14 MSun" % (massLimit_90Complete[np.argmin(abs(zBinCentres-0.5))]))
        print("... survey-averaged 90%% mass completeness limit (0.2 < z < 1.0) = %.1f x 10^14 MSun" % (averageMassLimit_90Complete))

#------------------------------------------------------------------------------------------------------------
def calcCompletenessContour(compMz, log10M, z, level = 0.90):
    """Calculates a completeness contour on the (log\ :sub:`10` mass, z) plane.
    
    Args:
        compMz (:obj:`np.ndarray`): Map (2d array) of completeness on the (log\ :sub:`10` mass, z) plane.
        log10M (:obj:`np.ndarray`): One dimensional array of log\ :sub:`10` mass values corresponding to
            `compMz`.
        z (:obj:`np.ndarray`): One dimensional arra of redshifts corresponding to `compMz`.
        level (:obj:`float`, optional): Fractional completeness level (e.g., 0.90 is 90% completeness).
    
    Returns:
        Contour values for the given completeness level (a pair of arrays - redshifts, and log\ :sub:`10`
        mass values).
    
    """
    # Easiest way to get at contour for plotting later
    # The smoothing may only be necessary if compMz is made by montecarlo method
    contours=plt.contour(z, log10M, compMz.transpose(), levels = [level])
    cont_z=[]
    cont_log10M=[]
    for p in contours.collections[0].get_paths():
        v=p.vertices
        cont_z=cont_z+v[:, 0].tolist()
        cont_log10M=cont_log10M+v[:, 1].tolist()
    plt.close()
    contTab=atpy.Table()
    contTab.add_column(atpy.Column(cont_z, 'z'))
    contTab.add_column(atpy.Column(cont_log10M, 'log10M'))
    contTab.sort('z')
    cont_z=[]
    cont_log10M=[]
    for zi in z:
        mask=np.equal(contTab['z'], zi)
        if mask.sum() > 0:
            cont_z.append(zi)
            cont_log10M.append(np.median(contTab['log10M'][mask]))
    cont_z=np.array(cont_z)
    cont_log10M=np.array(cont_log10M)
    #cont_log10M=ndimage.uniform_filter(cont_log10M, int(np.ceil(len(cont_log10M)/20)))
    
    return cont_z, cont_log10M
    
#------------------------------------------------------------------------------------------------------------
def makeMzCompletenessPlot(compMz, log10M, z, title, massLabel, outFileName):
    """Makes a (log\ :sub:`10` mass, z) completeness plot.
    
    Args:
        compMz (:obj:`np.ndarray`): Map (2d array) of completeness on the (log\ :sub:`10` mass, z) plane.
        log10M (:obj:`np.ndarray`): One dimensional array of log\ :sub:`10` mass values corresponding to
            `compMz`.
        z (:obj:`np.ndarray`): One dimensional arra of redshifts corresponding to `compMz`.
        title (:obj:`str`): Title that will be written at the top of the plot.
        massLabel (:obj:`str`): Label for mass quantity (e.g., "M200c").
        outFileName (:obj:`str`): Path where the plot will be written to a file, with the format being
            determined by the file extension.
    
    Returns:
        None
        
    """
    
    cont_z, cont_log10M=calcCompletenessContour(compMz, log10M, z, level = 0.90)
    
    # Actual plot
    plotSettings.update_rcParams()
    fig, ax = plt.subplots(figsize=(9.5,6.5))

    plt.imshow((compMz*100).transpose(), cmap = colorcet.m_rainbow, origin = 'lower', aspect = 'auto')
    
    y_tck=interpolate.splrep(log10M, np.arange(log10M.shape[0]))
    plot_log10M=np.linspace(13.5, 15.5, 9)
    coords_log10M=interpolate.splev(plot_log10M, y_tck)
    labels_log10M=[]
    for lm in plot_log10M:
        labels_log10M.append("%.2f" % (lm))
    plt.yticks(interpolate.splev(plot_log10M, y_tck), labels_log10M)
    plt.ylim(coords_log10M.min(), coords_log10M.max())
    if massLabel[0] == "M":
        massLabel=massLabel[1:]
    plt.ylabel("log$_{10}$ ($M_{\\rm %s} / M_{\odot}$)" % (massLabel))
    
    x_tck=interpolate.splrep(z, np.arange(z.shape[0]))
    plot_z=np.linspace(0.0, 2.0, 11)
    coords_z=interpolate.splev(plot_z, x_tck)
    labels_z=[]
    for lz in plot_z:
        labels_z.append("%.1f" % (lz))
    plt.xticks(interpolate.splev(plot_z, x_tck), labels_z)
    plt.xlim(coords_z.min(), coords_z.max())
    plt.xlabel("$z$")
    
    coords_cont_z=interpolate.splev(cont_z, x_tck)
    coords_cont_log10M=interpolate.splev(cont_log10M, y_tck)
    plt.plot(coords_cont_z, coords_cont_log10M, 'k:', lw = 3)
    
    plt.colorbar(pad = 0.03)
    cbLabel="Completeness (%)" 
    plt.figtext(0.96, 0.52, cbLabel, ha="center", va="center", family = "sans-serif", rotation = "vertical")

    plt.tight_layout()

    if title != 'full':
        plt.title(title)    
    plt.savefig(outFileName)
    plt.close()

#------------------------------------------------------------------------------------------------------------
def calcMassLimit(completenessFraction, compMz, mockSurvey, zBinEdges = []):
    """Given a completeness (log\ :sub:`10` mass, z) grid as made by :meth:`calcCompleteness`, return the
    mass limit (units of 10\ :sup:`14` M\ :sub:`Sun`) as a function of redshift at the given completeness
    level. By default, the same binning as the given `mockSurvey` object is used - this can be overridden by
    giving `zBinEdges`.
    
    Args:
        completenessFraction (:obj:`float`): Fractional completeness level (e.g., 0.90 is 90% completeness).
        compMz (:obj:`np.ndarray`): Map (2d array) of completeness on the (log\ :sub:`10` mass, z) plane.
        mockSurvey (:class:`nemo.MockSurvey.MockSurvey`): A :class:`MockSurvey` object, used for halo mass
            function calculations and generating mock catalogs.
        zBinEdges (:obj:`np.ndarray`, optional): Redshifts at which the mass limit is evaluated.

    Returns:
        The mass limit (units of 10\ :sup:`14` M\ :sub:`Sun`, 1d array) at each requested redshift.
    
    """
    
    massLimit=np.power(10, mockSurvey.log10M[np.argmin(abs(compMz-completenessFraction), axis = 1)])/1e14
    
    if zBinEdges != []:
        binnedMassLimit=np.zeros(len(zBinEdges)-1)
        for i in range(len(zBinEdges)-1):
            binnedMassLimit[i]=np.average(massLimit[np.logical_and(mockSurvey.z > zBinEdges[i], mockSurvey.z <= zBinEdges[i+1])])
        massLimit=binnedMassLimit
    
    return massLimit
    
#------------------------------------------------------------------------------------------------------------
def calcCompleteness(RMSTab, SNRCut, tileName, mockSurvey, massOptions, QFit, plotFileName = None, z = None,
                     method = "fast", numDraws = 2000000, numIterations = 100, verbose = False):
    """Calculate completeness as a function of (log\ :sub:`10` mass, z) on the mockSurvey grid at the given
    `SNRCut`. Intrinsic scatter in the scaling relation is taken into account.
    
    Args:
        RMSTab (:obj:`astropy.table.Table`): Table containing noise level by area, as returned by
            :meth:`getRMSTab`.
        SNRCut (:obj:`float`): Completeness will be calculated for objects relative to this cut in 
            ``fixed_SNR``.
        tileName (:obj:`str`): Name of the map tile.
        mockSurvey (:class:`nemo.MockSurvey.MockSurvey`): A :class:`MockSurvey` object, used for halo mass
            function calculations and generating mock catalogs.
        massOptions (:obj:`dict`): A dictionary of scaling relation, cosmological, and mass definition
            parameters (see example Nemo config files for the format).
        QFit (:class:`nemo.signals.QFit`): An object for calculating the filter mismatch function, referred
            to as `Q` in the ACT papers from `Hasselfield et al. (2013) <http://adsabs.harvard.edu/abs/2013JCAP...07..008H>`_
            onwards.
        plotFileName (:obj:`str`, optional): If given, write a plot showing 90% completness limit to this
            path.
        z (:obj:`float`, optional): Redshift at which the completeness calculation will be performed.
            If None, the redshift range will be taken from the :class:`MockSurvey` object.
        method (:obj:`str`, optional): Two methods for doing the calculation are available: "fast" (applies
            the measurement errors and scatter to 'true' ỹ\ :sub:`0` values on a grid) and "montecarlo" (uses
            samples drawn from a mock catalog, generated on the fly, to estimate the completeness). Both
            methods should give consistent results.
        numDraws (:obj:`int`, optional): Used by the "montecarlo" method - sets the number of draws from the
            halo mass function on each iteration.
        numIterations (:obj:`int`, optional): Used by the "montecarlo" method - sets the number of iterations,
            i.e., the number of mock catalogs from which the completeness is estimated.

    Returns:
        A 2d array of (log\ :sub:`10` mass, z) completeness.
    
    """
        
    if z is not None:
        zIndex=np.argmin(abs(mockSurvey.z-z))
        zRange=mockSurvey.z[zIndex:zIndex+1]
    else:
        zRange=mockSurvey.z

    trueMassCol="true_M%d%s" % (mockSurvey.delta, mockSurvey.rhoType[0])

    if verbose == True:
        print("... calcuating completeness in tile %s using '%s' method" % (tileName, method))

    if method == "montecarlo":

        #t0=time.time()
        # Need area-weighted average noise in the tile - we could change this to use entire RMS map instead
        areaWeights=RMSTab['areaDeg2'].data/RMSTab['areaDeg2'].data.sum()
        if areaWeights.sum() > 0:
            y0Noise=np.average(RMSTab['y0RMS'].data, weights = areaWeights)
            # Monte-carlo sims approach: slow - but can use to verify the other approach below
            halfBinWidth=(mockSurvey.log10M[1]-mockSurvey.log10M[0])/2.0
            binEdges_log10M=(mockSurvey.log10M-halfBinWidth).tolist()+[np.max(mockSurvey.log10M)+halfBinWidth]
            halfBinWidth=(mockSurvey.z[1]-mockSurvey.z[0])/2.0
            binEdges_z=(zRange-halfBinWidth).tolist()+[np.max(zRange)+halfBinWidth]
            allMz=np.zeros([mockSurvey.clusterCount.shape[1], mockSurvey.clusterCount.shape[0]])
            detMz=np.zeros([mockSurvey.clusterCount.shape[1], mockSurvey.clusterCount.shape[0]])
            for i in range(numIterations):
                tab=mockSurvey.drawSample(y0Noise, massOptions, QFit, tileName = tileName,
                                          SNRLimit = SNRCut, applySNRCut = False, z = z, numDraws = numDraws,
                                          applyRelativisticCorrection = massOptions['relativisticCorrection'])
                allMz=allMz+np.histogram2d(np.log10(tab[trueMassCol]*1e14), tab['redshift'], [binEdges_log10M, binEdges_z])[0]
                detMask=np.greater(tab['fixed_y_c']*1e-4, y0Noise*SNRCut)
                detMz=detMz+np.histogram2d(np.log10(tab[trueMassCol][detMask]*1e14), tab['redshift'][detMask], [binEdges_log10M, binEdges_z])[0]
            mask=np.not_equal(allMz, 0)
            compMz=np.ones(detMz.shape)
            compMz[mask]=detMz[mask]/allMz[mask]
            compMz=compMz.transpose()
        else:
            compMz=np.zeros([mockSurvey.clusterCount.shape[0], mockSurvey.clusterCount.shape[1]])
        #astImages.saveFITS("test_compMz_MC_5000.fits", compMz.transpose(), None)
        #t1=time.time()

    elif method == "fast":

        # Using full noise distribution, weighted by fraction of area
        # NOTE: removed recMassBias and div parameters
        tenToA0, B0, Mpivot, sigma_int=[massOptions['tenToA0'], massOptions['B0'],
                                        massOptions['Mpivot'], massOptions['sigma_int']]
        y0Grid=np.zeros([zRange.shape[0], mockSurvey.clusterCount.shape[1]])
        for i in range(len(zRange)):
            zk=zRange[i]
            k=np.argmin(abs(mockSurvey.z-zk))
            # WARNING: the interpolates for theta500, fRel in MockSurvey work in terms of M500c
            # mockSurvey.log10M is NOT necessarily M500c any more
            if massOptions['delta'] == 500 and massOptions['rhoType'] == "critical":
                log10M500cs=mockSurvey.log10M
            else:
                log10M500cs=np.log10(mockSurvey.mdef.translate_mass(mockSurvey.cosmoModel, np.power(10, mockSurvey.log10M),
                                                                    1/(1+zk), mockSurvey._M500cDef))

            theta500s_zk=interpolate.splev(log10M500cs, mockSurvey.theta500Splines[k])
            Qs_zk=QFit.getQ(theta500s_zk, z = zk, tileName = tileName)
            true_y0s_zk=tenToA0*np.power(mockSurvey.Ez[k], 2)*np.power(np.power(10, mockSurvey.log10M)/Mpivot, 1+B0)*Qs_zk
            if massOptions['relativisticCorrection'] == True:
                fRels_zk=interpolate.splev(log10M500cs, mockSurvey.fRelSplines[k])
                true_y0s_zk=true_y0s_zk*fRels_zk
            y0Grid[i]=true_y0s_zk
            
        # For some cosmological parameters, we can still get the odd -ve y0
        y0Grid[y0Grid <= 0] = 1e-9
        
        # Calculate completeness using area-weighted average
        # NOTE: RMSTab that is fed in here can be downsampled in noise resolution for speed
        areaWeights=RMSTab['areaDeg2']/RMSTab['areaDeg2'].sum()
        log_y0Lim=np.log(SNRCut*RMSTab['y0RMS'])
        log_y0=np.log(y0Grid)
        compMz=np.zeros(log_y0.shape)
        for i in range(len(RMSTab)):
            SNRGrid=y0Grid/RMSTab['y0RMS'][i]
            SNRGrid=SNRGrid
            log_y0Err=1/SNRGrid
            log_y0Err[SNRGrid < SNRCut]=1/SNRCut
            log_totalErr=np.sqrt(log_y0Err**2 + sigma_int**2)
            compMz=compMz+stats.norm.sf(log_y0Lim[i], loc = log_y0, scale = log_totalErr)*areaWeights[i]
        
        #t1=time.time()
        
        # For checking figure-of-merit
        #predMz=compMz*mockSurvey.clusterCount
        #predMz=predMz/predMz.sum()
        #astImages.saveFITS("predMz.fits", predMz.transpose(), None)        
        #projImg=pyfits.open("projMz_SNR%.2f.fits" % (SNRCut))        
        #projMz=projImg[0].data.transpose()
        #projImg.close()
        #merit=np.sum(np.sqrt(np.power(projMz-predMz, 2)))
        #print(merit)
        #IPython.embed()
        #sys.exit()
            
    elif method == "injection":
        print("calc completeness from injection sim")
        import IPython
        IPython.embed()
        sys.exit()

    elif method is None:
        return None

    else:
        raise Exception("calcCompleteness only has 'fast', and 'Monte Carlo' methods available")
            
    if plotFileName is not None:
        # Calculate 90% completeness as function of z
        zBinEdges=np.arange(0.05, 2.1, 0.1)
        zBinCentres=(zBinEdges[:-1]+zBinEdges[1:])/2.
        massLimit_90Complete=calcMassLimit(0.9, compMz, mockSurvey, zBinEdges = zBinEdges)
        zMask=np.logical_and(zBinCentres >= 0.2, zBinCentres < 1.0)
        averageMassLimit_90Complete=np.average(massLimit_90Complete[zMask])
        makeMassLimitVRedshiftPlot(massLimit_90Complete, zBinCentres, plotFileName, 
                                   title = "%s: $M_{\\rm %d%s}$ / $10^{14}$ M$_{\odot}$ > %.2f (0.2 < $z$ < 1)" % (tileName,
                                   mockSurvey.delta, mockSurvey.rhoType[0], averageMassLimit_90Complete))
            
    return compMz
      
#------------------------------------------------------------------------------------------------------------
def makeMassLimitMap(RMSTab, SNRCut, z, tileName, photFilterLabel, mockSurvey, scalingRelationDict, QFit,
                     diagnosticsDir, selFnDir):
    """Makes a map of 90% mass completeness (for now, this fraction is fixed and not adjustable, and the full
    survey footprint is used). The map is written as a FITS file in the `diagnosticsDir` directory.
    
    Args:
        RMSTab (:obj:`astropy.table.Table`): Table of RMS values for the tile, as produced by :meth:`getRMSTab`.
        SNRCut (:obj:`float`): Completeness will be calculated for objects relative to this cut in
            ``fixed_SNR``.
        z (:obj:`float`): The redshift at which the mass limit map will be produced.
        tileName (:obj:`str`): The name of the tile.
        photFilterLabel (:obj:`str`): Name of the reference filter, as specified in, e.g., a 
            :ref:`nemoCommand` config file (see :ref:`ConfigReference`).
        mockSurvey (:class:`nemo.MockSurvey.MockSurvey`): A :class:`MockSurvey` object, used for halo mass function 
            calculations and generating mock catalogs.
        scalingRelationDict (:obj:`dict`): A dictionary of scaling relation parameters (see example Nemo
            config files for the format).
        QFit (:class:`nemo.signals.QFit`): An object for calculating the filter mismatch function, referred
            to as `Q` in the ACT papers from `Hasselfield et al. (2013) <http://adsabs.harvard.edu/abs/2013JCAP...07..008H>`_
            onwards.
        diagnosticsDir (:obj:`str`): Path to the ``diagnostics/`` directory, as produced by the 
            :ref:`nemoCommand` command.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
    
    """
        
    RMSMap, wcs=loadRMSMap(tileName, selFnDir, photFilterLabel)

    # Fill in blocks in map for each RMS value
    outFileName=diagnosticsDir+os.path.sep+tileName+os.path.sep+"massLimitMap_z%s#%s.fits" % (str(z).replace(".", "p"), tileName)
    if os.path.exists(outFileName) == False:
        massLimMap=np.zeros(RMSMap.shape)
        count=0
        t0=time.time()
        for y0Noise in RMSTab['y0RMS']:
            count=count+1
            compMz=calcCompleteness(RMSTab[np.where(RMSTab['y0RMS'] == y0Noise)], SNRCut, tileName, mockSurvey, 
                                    scalingRelationDict, QFit, z = z)
            massLimMap[RMSMap == y0Noise]=mockSurvey.log10M[np.argmin(abs(compMz-0.9))]
        t1=time.time()
        mask=np.not_equal(massLimMap, 0)
        massLimMap[mask]=np.power(10, massLimMap[mask])/1e14
        maps.saveFITS(outFileName, massLimMap, wcs, compressionType = "RICE_1")
        
#------------------------------------------------------------------------------------------------------------
def makeMassLimitVRedshiftPlot(massLimit_90Complete, zRange, outFileName, title = None):
    """Makes a plot of 90%-completeness mass limit versus redshift. Uses spline interpolation.
    
    Args:
        massLimit_90Complete (:obj:`np.ndarray`): Mass limit at each redshift, corresponding to 90%
            completeness.
        zRange (:obj:`np.ndarray`): Redshifts at which mass completeness was evaluted.
        outFileName (:obj:`str`): Path to which the plot file will be written. The format is determined by
            the file extension.
        title (:obj:`str`): The title that will be written at the top of the plot.
    
    Returns:
        None
    
    """
    
    plotSettings.update_rcParams()
    plt.figure(figsize=(9,6.5))
    ax=plt.axes([0.10, 0.11, 0.87, 0.86])
    if title is not None:
        plt.figtext(0.15, 0.2, title, ha="left", va="center")
    tck=interpolate.splrep(zRange, massLimit_90Complete)
    plotRange=np.linspace(0, 2, 100)
    plt.plot(plotRange, interpolate.splev(plotRange, tck), 'k-')
    plt.plot(zRange, massLimit_90Complete, 'D', ms = 8)
    plt.xlabel("$z$")
    plt.ylim(0.5, 8)
    plt.xticks(np.arange(0, 2.2, 0.2))
    plt.xlim(0, 2)
    labelStr="$M_{\\rm 500c}$ (10$^{14}$ M$_{\odot}$) [90% complete]"
    plt.ylabel(labelStr)
    plt.savefig(outFileName)
    if outFileName.find(".pdf") != -1:
        plt.savefig(outFileName.replace(".pdf", ".png"))
    plt.close()   
    
#------------------------------------------------------------------------------------------------------------
def cumulativeAreaMassLimitPlot(z, diagnosticsDir, selFnDir, tileNames):
    """Makes cumulative plots of the 90%-completeness mass limit versus survey area, at the given redshift.
    
    Args:
        z (:obj:`float`): The redshift at which the mass completeness is evaluated.
        diagnosticsDir (:obj:`str`): Path to the ``diagnostics/`` directory, as produced by the 
            :ref:`nemoCommand` command.
        selFnDir (:obj:`str`): Path to a ``selFn/`` directory, as produced by the :ref:`nemoCommand`
            command. This directory contains information such as the survey noise maps, area masks,
            and information needed to construct the filter mismatch function, `Q`, used in mass
            modeling.
        tileNames (:obj:`list`): List of tiles to use.
    
    Note:
        This routine writes plots into the `diagnosticsDir` directory (one for the cumulative mass
        completeness limit over the whole survey area, and one for the deepest 20% only).
    
    Returns:
        None
        
    """
        
    # NOTE: We truncate mass limits to 0.1 level here - differences beyond that are due to compression
    allLimits=[]
    allAreas=[]
    for tileName in tileNames:
        massLimMap, wcs=loadMassLimitMap(tileName, diagnosticsDir+os.path.sep+tileName, z)
        areaMap, wcs=loadAreaMask(tileName, selFnDir)
        areaMapSqDeg=(maps.getPixelAreaArcmin2Map(areaMap.shape, wcs)*areaMap)/(60**2)
        limits=np.unique(massLimMap).tolist()
        if limits[0] == 0:
            limits=limits[1:]
        areas=[]
        truncLim=[]
        for l in limits:
            truncLim.append(round(l, 1))
            #truncLim.append(float(Decimal(l).quantize(Decimal('0.1'))))
            areas.append(areaMapSqDeg[np.where(massLimMap == l)].sum())
        allLimits=allLimits+truncLim
        allAreas=allAreas+areas
    
    # Reduce redundant mass limits
    allLimits=np.array(allLimits)
    allAreas=np.array(allAreas)
    uniqLimits=np.unique(allLimits)
    uniqAreas=[]
    for u in uniqLimits:
        uniqAreas.append(allAreas[allLimits == u].sum())
    uniqAreas=np.array(uniqAreas)
    tab=atpy.Table()
    tab.add_column(atpy.Column(uniqLimits, 'MLim'))
    tab.add_column(atpy.Column(uniqAreas, 'areaDeg2'))
    tab.sort('MLim')
        
    plotSettings.update_rcParams()
        
    # Full survey plot
    plt.figure(figsize=(9,6.5))
    ax=plt.axes([0.155, 0.12, 0.82, 0.86])
    plt.minorticks_on()
    plt.plot(tab['MLim'], np.cumsum(tab['areaDeg2']), 'k-')
    #plt.plot(plotMRange, plotCumArea, 'k-')
    plt.ylabel("survey area < $M_{\\rm 500c}$ limit (deg$^2$)")
    plt.xlabel("$M_{\\rm 500c}$ (10$^{14}$ M$_{\odot}$) [90% complete]")
    labelStr="total survey area = %.0f deg$^2$" % (np.cumsum(tab['areaDeg2']).max())
    plt.ylim(0.0, 1.2*np.cumsum(tab['areaDeg2']).max())
    plt.xlim(tab['MLim'].min(), tab['MLim'].max())
    plt.figtext(0.2, 0.9, labelStr, ha="left", va="center")
    plt.savefig(diagnosticsDir+os.path.sep+"cumulativeArea_massLimit_z%s.pdf" % (str(z).replace(".", "p")))
    plt.savefig(diagnosticsDir+os.path.sep+"cumulativeArea_massLimit_z%s.png" % (str(z).replace(".", "p")))
    plt.close()
    
    # Deepest 20% - we show a bit beyond this
    totalAreaDeg2=tab['areaDeg2'].sum()
    deepTab=tab[np.where(np.cumsum(tab['areaDeg2']) < 0.25 * totalAreaDeg2)]
    plt.figure(figsize=(9,6.5))
    ax=plt.axes([0.155, 0.12, 0.82, 0.86])
    plt.minorticks_on()
    plt.plot(tab['MLim'], np.cumsum(tab['areaDeg2']), 'k-')
    plt.ylabel("survey area < $M_{\\rm 500c}$ limit (deg$^2$)")
    plt.xlabel("$M_{\\rm 500c}$ (10$^{14}$ M$_{\odot}$) [90% complete]")
    labelStr="area of deepest 20%% = %.0f deg$^2$" % (0.2 * totalAreaDeg2)
    plt.ylim(0.0, 1.2*np.cumsum(deepTab['areaDeg2']).max())
    plt.xlim(deepTab['MLim'].min(), deepTab['MLim'].max())
    plt.figtext(0.2, 0.9, labelStr, ha="left", va="center")
    plt.savefig(diagnosticsDir+os.path.sep+"cumulativeArea_massLimit_z%s_deepest20Percent.pdf" % (str(z).replace(".", "p")))
    plt.savefig(diagnosticsDir+os.path.sep+"cumulativeArea_massLimit_z%s_deepest20Percent.png" % (str(z).replace(".", "p")))
    plt.close()
        
#------------------------------------------------------------------------------------------------------------
def makeFullSurveyMassLimitMapPlot(z, config):
    """Makes full area mass limit map by reprojecting the tile mass limit maps onto the full map pixelisation.
    Creates both a plot and a FITS image.
    
    Args:
        z (:obj:`float`): The redshift at which the mass completeness is evaluated.
        config (:class:`nemo.startUp.NemoConfig`): A NemoConfig object.
    
    Note:
        Output is written to the `diagnostics` directory, as produced by the :ref:`nemoCommand` command.
    
    Returns:
        None
    
    """
    
    if 'makeQuickLookMaps' not in config.parDict.keys():
        config.quicklookScale=0.25
        config.quicklookShape, config.quicklookWCS=maps.shrinkWCS(config.origShape, config.origWCS, config.quicklookScale)

    outFileName=config.diagnosticsDir+os.path.sep+"reproj_massLimitMap_z%s.fits" % (str(z).replace(".", "p"))
    maps.stitchTilesQuickLook(config.diagnosticsDir+os.path.sep+"*"+os.path.sep+"massLimitMap_z%s#*.fits" % (str(z).replace(".", "p")),
                              outFileName, config.quicklookWCS, config.quicklookShape,
                              fluxRescale = config.quicklookScale)

    # Make plot
    if os.path.exists(outFileName) == True:
        with pyfits.open(outFileName) as img:
            for hdu in img:
                if hdu.shape != ():
                    reproj=np.nan_to_num(hdu.data)
                    reproj=np.ma.masked_where(reproj <1e-6, reproj)
                    wcs=astWCS.WCS(hdu.header, mode = 'pyfits')
            plotSettings.update_rcParams()
            fontSize=20.0
            figSize=(16, 5.7)
            axesLabels="sexagesimal"
            axes=[0.08,0.15,0.91,0.88]
            cutLevels=[2, int(np.median(reproj[np.nonzero(reproj)]))+2]
            colorMapName=colorcet.m_rainbow
            fig=plt.figure(figsize = figSize)
            p=astPlots.ImagePlot(reproj, wcs, cutLevels = cutLevels, title = None, axes = axes, 
                                axesLabels = axesLabels, colorMapName = colorMapName, axesFontFamily = 'sans-serif', 
                                RATickSteps = {'deg': 30.0, 'unit': 'h'}, decTickSteps = {'deg': 20.0, 'unit': 'd'},
                                axesFontSize = fontSize)
            cbLabel="$M_{\\rm 500c}$ (10$^{14}$ M$_{\odot}$) [90% complete]"
            cbShrink=0.7
            cbAspect=40
            cb=plt.colorbar(p.axes.images[0], ax = p.axes, orientation="horizontal", fraction = 0.05, pad = 0.18, 
                            shrink = cbShrink, aspect = cbAspect)
            plt.figtext(0.53, 0.04, cbLabel, ha="center", va="center", fontsize = fontSize, family = "sans-serif")
            plt.savefig(outFileName.replace(".fits", ".pdf"), dpi = 300)
            plt.savefig(outFileName.replace(".fits", ".png"), dpi = 300)
            plt.close()

#------------------------------------------------------------------------------------------------------------
def tidyUp(config):
    """Tidies up the `selFn` directory, constructing multi-extension FITS files from individual tile images
    and tables, deleting the individual tile files when complete. This routine also copies the given Nemo
    configuration into the `selFn` directory, and writes a plain text file that lists the tile names, and the
    areas covered by each tile.
    
    Args:
        config (:class:`nemo.startUp.NemoConfig`): A NemoConfig object.
        
    Returns:
        None
    
    """
    
    shutil.copy(config.configFileName, config.selFnDir+os.path.sep+"config.yml")

    # Make MEFs
    MEFsToBuild=["RMSMap_%s" % (config.parDict['photFilter'])]
    compressionTypes=["RICE_1"]
    dtypes=[np.float]
    if 'selFnFootprints' in config.parDict.keys():
        for footprintDict in config.parDict['selFnFootprints']:
            MEFsToBuild.append("intersect_%s" % footprintDict['label'])
            compressionTypes.append("PLIO_1")
            dtypes.append(np.uint8)
    assert(len(MEFsToBuild) == len(compressionTypes))
    assert(len(MEFsToBuild) == len(dtypes))
    for MEFBaseName, compressionType, dtype in zip(MEFsToBuild, compressionTypes, dtypes):
        outFileName=config.selFnDir+os.path.sep+MEFBaseName+".fits"
        newImg=pyfits.HDUList()
        filesToRemove=[]
        for tileName in config.allTileNames:
            fileName=config.selFnDir+os.path.sep+tileName+os.path.sep+MEFBaseName+"#"+tileName+".fits"
            if os.path.exists(fileName):
                with pyfits.open(fileName) as img:
                    for extName in img:
                        if img[extName].data is not None:
                            break
                    hdu=pyfits.CompImageHDU(np.array(img[extName].data, dtype = dtype), img[extName].header, 
                                            name = tileName, compression_type = compressionType)
                filesToRemove.append(fileName)
                newImg.append(hdu)
        if len(newImg) > 0:
            newImg.writeto(outFileName, overwrite = True)
            for f in filesToRemove:
                os.remove(f)

    # Write a table of tile areas for those that want it
    with open(config.selFnDir+os.path.sep+"tileAreas.txt", "w") as outFile:
        outFile.write("#tileName areaDeg2\n")
        for tileName in config.allTileNames:
            tileAreaDeg2=getTileTotalAreaDeg2(tileName, config.selFnDir)
            outFile.write("%s %.6f\n" % (tileName, tileAreaDeg2))

    # Clean out any per-tile directories
    for tileName in config.allTileNames:
        if os.path.exists(config.selFnDir+os.path.sep+tileName) == True:
            shutil.rmtree(config.selFnDir+os.path.sep+tileName)


    
