"""

Tools used by nemoSelFn

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
from scipy import ndimage
from scipy import optimize
from nemo import simsTools
from nemo import mapTools
from nemo import MockSurvey
from nemo import plotSettings
import types
import pickle
import astropy.io.fits as pyfits
import time
import IPython
plt.matplotlib.interactive(False)

# If want to catch warnings as errors...
#import warnings
#warnings.filterwarnings('error')

#------------------------------------------------------------------------------------------------------------
def getTileTotalAreaDeg2(extName, diagnosticsDir):
    """Returns total area of the tile pointed at by extName (taking into account survey mask and point
    source masking).
    
    """
    
    areaImg=pyfits.open(diagnosticsDir+os.path.sep+"areaMask#%s.fits" % (extName))
    areaMap=areaImg[0].data
    wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
    areaMapSqDeg=(mapTools.getPixelAreaArcmin2Map(areaMap, wcs)*areaMap)/(60**2)
    totalAreaDeg2=areaMapSqDeg.sum()
    
    return totalAreaDeg2

#------------------------------------------------------------------------------------------------------------
def getRMSTab(extName, photFilterLabel, diagnosticsDir):
    """Makes a table containing fraction of map area in tile pointed to by extName against RMS values
    (so this compresses the information in the RMS maps). The first time this is run takes ~200 sec (for a
    1000 sq deg tile), but the result is cached.
        
    Returns RMSTab
    
    """
    
    # This can probably be sped up, but takes ~200 sec for a ~1000 sq deg tile, so we cache
    RMSTabFileName=diagnosticsDir+os.path.sep+"RMSTab_%s.fits" % (extName)
    if os.path.exists(RMSTabFileName) == False:
        print(("... making %s ..." % (RMSTabFileName)))
        RMSImg=pyfits.open(diagnosticsDir+os.path.sep+"RMSMap_Arnaud_M2e14_z0p4#%s.fits" % (extName))
        RMSMap=RMSImg[0].data
        RMSValues=np.unique(RMSMap[np.nonzero(RMSMap)])

        areaImg=pyfits.open(diagnosticsDir+os.path.sep+"areaMask#%s.fits" % (extName))
        areaMap=areaImg[0].data
        wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
        areaMapSqDeg=(mapTools.getPixelAreaArcmin2Map(areaMap, wcs)*areaMap)/(60**2)
        totalAreaDeg2=areaMapSqDeg.sum()
        
        fracArea=np.zeros(len(RMSValues))
        for i in range(len(RMSValues)):
            fracArea[i]=areaMapSqDeg[np.equal(RMSMap, RMSValues[i])].sum()
        RMSTab=atpy.Table()
        RMSTab.add_column(atpy.Column(fracArea, 'fracArea'))
        RMSTab.add_column(atpy.Column(RMSValues, 'y0RMS'))
        RMSTab.write(RMSTabFileName)
    else:
        #print("... reading %s ..." % (RMSTabFileName))
        RMSTab=atpy.Table().read(RMSTabFileName)
    
    return RMSTab

#------------------------------------------------------------------------------------------------------------
def calcTileWeightedAverageNoise(extName, photFilterLabel, diagnosticsDir):
    """Returns weighted average noise value in the tile.
    
    """

    RMSTab=getRMSTab(extName, photFilterLabel, diagnosticsDir)
    RMSValues=np.array(RMSTab['y0RMS'])
    fracArea=np.array(RMSTab['fracArea'])
    tileRMSValue=np.average(RMSValues, weights = fracArea)

    return tileRMSValue

#------------------------------------------------------------------------------------------------------------
def rayleighFlipped(log10M, loc, scale):
    """This is a good functional form to fit to completeness as a function of mass. It's asymmetric, so
    it can fit both the low and high completeness ends with one function (previously we used Gaussian cdf,
    but that isn't a good fit when intrinsic scatter is turned on).
    
    """
    return (1-stats.rayleigh.cdf(log10M, loc = loc, scale = scale))[::-1]

#------------------------------------------------------------------------------------------------------------
def calcCompleteness(y0Noise, SNRCut, extName, mockSurvey, scalingRelationDict, tckQFitDict, diagnosticsDir,
                     zRange = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0],
                     fitTabFileName = None):
    """Calculate completeness as a function of M500, z, for assumed fiducial cosmology and scaling relation,
    at the given SNRCut and noise level. Intrinsic scatter in the scaling relation is taken into account.
    
    If fitTabFileName != None, saves the resulting model (and sim results) as .fits table(s) under 
    diagnosticsDir. Also saves a diagnostic plot of completeness for that tile.
    
    Returns fitTab
    
    """
    
    if fitTabFileName != None and os.path.exists(fitTabFileName) == True:
        fitTab=atpy.Table().read(fitTabFileName)
        return fitTab
    
    # What we'll generally want is the completeness at some fixed z, for some assumed scaling relation
    tenToA0, B0, Mpivot, sigma_int=[scalingRelationDict['tenToA0'], scalingRelationDict['B0'], 
                                    scalingRelationDict['Mpivot'], scalingRelationDict['sigma_int']]

    # This is the minimum number to draw - gets multiplied up as needed to get good statistics...
    # ... and this is the minimum number of detected sim cluster we need to get a good completeness curve and model fit
    # This combination peaks at ~5 Gb of RAM used 
    numMockClusters=10000
    minAllowedDetections=4000
    maxMultiplier=8000
    minDrawNoiseMultiplier=SNRCut-3.  # This is only for an estimate, but can make a huge difference in speed if set higher (not always robust though)

    # Binning used for fitting
    binEdges=np.arange(13.5, 16.0, 0.05)
    binCentres=(binEdges[:-1]+binEdges[1:])/2.

    # For storing results - this one is not strictly necessary, but could be used for sanity checking fit results
    # Table format: column of log10M values, then columns (labelled with redshift) of completeness versus mass
    completenessTab=atpy.Table()
    completenessTab.add_column(atpy.Column(binCentres, 'log10M'))

    t00=time.time()

    # Stores Rayleigh cumulative distribution function (flipped etc.) fits at each z
    fitTab=atpy.Table()
    fitTab.add_column(atpy.Column(zRange, 'z'))
    fitTab.add_column(atpy.Column(np.zeros(len(zRange)), 'loc'))
    fitTab.add_column(atpy.Column(np.zeros(len(zRange)), 'scale'))

    # Need these when applying the fit - only valid in this range for given loc, scale
    fitTab.add_column(atpy.Column(np.zeros(len(zRange)), 'log10MMin'))  
    fitTab.add_column(atpy.Column(np.zeros(len(zRange)), 'log10MMax'))

    # We may as well store 90% completeness limit in here as well
    fraction=0.9
    fitTab.add_column(atpy.Column(np.zeros(len(zRange)), 'log10MLimit_90%'))

    for iz in range(len(zRange)):

        t0=time.time()    
        z=zRange[iz]
        #print("... z = %.2f ..." % (z))
        zIndex=np.where(abs(mockSurvey.z-z) == abs(mockSurvey.z-z).min())[0][0]
        
        # Used in Q-calc below...
        Ez=astCalc.Ez(z)
        Hz=astCalc.Ez(z)*astCalc.H0  
        G=4.301e-9  # in MSun-1 km2 s-2 Mpc
        criticalDensity=(3*np.power(Hz, 2))/(8*np.pi*G)
        
        # Ajusting mass limit for draws according to RMS - trick to avoid drawing loads of clusters we'll never see
        # This is ok as just a rough estimate
        minLog10MDraw=np.log10(Mpivot*np.power((y0Noise*minDrawNoiseMultiplier)/(tenToA0*np.power(astCalc.Ez(z), 2)), 1/(1+B0)))
        testDraws=np.linspace(0, 1, 1000000)
        testLog10M=interpolate.splev(testDraws, mockSurvey.tck_log10MRoller[zIndex])
        minDraw=testDraws[np.argmin(abs(testLog10M-minLog10MDraw))]
        if minDraw == 1.0:
            raise Exception("minDraw == 1 - very noisy pixel, try increasing number of test draws?")
            
        # Draw masses from the mass function...    
        numDetected=0
        mockClusterMultiplier=1
        exceededMultiplierCount=0
        while numDetected < minAllowedDetections:
            log10Ms=interpolate.splev(np.random.uniform(minDraw, 1, int(mockClusterMultiplier*numMockClusters)), mockSurvey.tck_log10MRoller[zIndex])
                        
            # Sort masses here, as we need in order biggest -> smallest for fast completeness calc
            log10Ms.sort()
            log10Ms=log10Ms[::-1]
            
            # For speedy mock "observations" - since we have fixed z
            fitM500s=np.power(10, np.linspace(log10Ms.min(), log10Ms.max(), 100))
            fitTheta500s=np.zeros(len(fitM500s))
            for i in range(len(fitM500s)):
                M500=fitM500s[i]
                R500Mpc=np.power((3*M500)/(4*np.pi*500*criticalDensity), 1.0/3.0)                     
                theta500Arcmin=np.degrees(np.arctan(R500Mpc/astCalc.da(z)))*60.0
                fitTheta500s[i]=theta500Arcmin
            tckLog10MToTheta500=interpolate.splrep(np.log10(fitM500s), fitTheta500s)
            theta500s=interpolate.splev(log10Ms, tckLog10MToTheta500)
            Qs=interpolate.splev(theta500s, tckQFitDict[extName])
            fRels=simsTools.calcFRel(z, np.power(10, log10Ms))
            true_y0s=tenToA0*np.power(astCalc.Ez(z), 2)*np.power(np.power(10, log10Ms)/Mpivot, 1+B0)*Qs*fRels
            
            # Mock "observations" (apply intrinsic scatter and noise)...
            scattered_y0s=np.exp(np.random.normal(np.log(true_y0s), sigma_int, len(true_y0s)))        
            measured_y0s=np.random.normal(scattered_y0s, y0Noise)
            # Comment out above and uncomment below to switch off intrinsic scatter
            #measured_y0s=np.random.normal(true_y0s, y0Noise)
            
            # Check selection - did we manage to select enough objects?
            y0Lim_selection=SNRCut*y0Noise  # y0Noise = RMS
            t111=time.time()
            try:
                numDetected=np.greater(measured_y0s, y0Lim_selection).sum()
            except:
                print("eh?")
                IPython.embed()
                sys.exit()
            if numDetected == 0:
                mockClusterMultiplier=maxMultiplier
            elif numDetected < minAllowedDetections:
                mockClusterMultiplier=1.2*(minAllowedDetections/(float(numDetected)/mockClusterMultiplier))
                #print("... mockClusterMultiplier = %.1f ..." % (mockClusterMultiplier))
                if mockClusterMultiplier > maxMultiplier:
                    mockClusterMultiplier=maxMultiplier
                    exceededMultiplierCount=exceededMultiplierCount+1
                if exceededMultiplierCount > 2:
                    raise Exception("exceeded maxMultiplier too many times")
        
        # Calculate completeness
        detArr=np.cumsum(np.greater(measured_y0s, y0Lim_selection))
        allArr=np.arange(1, len(log10Ms)+1, 1, dtype = float)
        completeness=detArr/allArr
        
        # Average/downsample: both for storage, and to deal with low numbers at high mass end (and spline fits later)
        binnedCompleteness=np.zeros(len(binEdges)-1)
        binnedCounts=np.zeros(len(binEdges)-1, dtype = float)
        for i in range(len(binEdges)-1):
            mask=np.logical_and(np.greater(log10Ms, binEdges[i]), np.less(log10Ms, binEdges[i+1]))
            if mask.sum() > 0:
                binnedCompleteness[i]=np.median(completeness[mask])
            binnedCounts[i]=mask.sum()
            # Never allow completness to decrease at higher mass (this would be a lower limit)
            if binnedCompleteness[i] < binnedCompleteness[i-1]:
                binnedCompleteness[i]=binnedCompleteness[i-1]
        completenessTab.add_column(atpy.Column(binnedCompleteness, z))
        
        # NOTE: Neither cdf nor error function can fit, because of intrinsic scatter (assymetric at low/high completeness)
        # However, cdf for Rayleigh distribution can be fiddled to look like we what we have...
        # We need the min, max of the range the fit is done over for finer binning to work (see below)
        fraction=0.5
        locGuess=log10Ms[np.argmin(abs(fraction-completeness))]
        fitResult=optimize.curve_fit(rayleighFlipped, binCentres, binnedCompleteness, p0 =[locGuess, 0.2])
        fittedLoc=fitResult[0][0]
        fittedScale=fitResult[0][1]
        fitTab['loc'][iz]=fittedLoc
        fitTab['scale'][iz]=fittedScale   
        fitTab['log10MMin'][iz]=binCentres.min()
        fitTab['log10MMax'][iz]=binCentres.max()

        # While we're here, we may as well store 90% completeness limit
        fraction=0.9    
        fineLog10Bins=np.linspace(fitTab['log10MMin'][iz], fitTab['log10MMax'][iz], 1000)
        fineCompleteness=rayleighFlipped(fineLog10Bins, fitTab['loc'][iz], fitTab['scale'][iz])
        log10MassLimit=fineLog10Bins[np.argmin(abs(fraction-fineCompleteness))]
        fitTab['log10MLimit_90%'][iz]=log10MassLimit
        #print("... 90%% completeness mass limit at z = %.1f: %.3e MSun ..." % (z, np.power(10, log10MassLimit)))
        t1=time.time()
        #print("... time taken = %.3f sec ..." % (t1-t0))

    t11=time.time()
    #print("... total time take for %s = %.3f sec ..." % (extName, t11-t00))
    
    if fitTabFileName != None:

        # Save both the fit results and the binned data they are based on
        outFileNames=[fitTabFileName, diagnosticsDir+os.path.sep+"selFn_completenessTab#%s.fits" % (extName)]
        tabs=[fitTab, completenessTab]
        for outFileName, tab in zip(outFileNames, tabs):
            if os.path.exists(outFileName) == True:
                os.remove(outFileName)
            tab.write(outFileName)
        
        # Tile-averaged 90% completeness mass limit and plot (will be a good, quick sanity check)
        massLimit_90Complete=np.power(10, np.array(fitTab['log10MLimit_90%']))/1e14
        zRange=np.array(fitTab['z'])
        averageMassLimit_90Complete=massLimit_90Complete[np.logical_and(np.greater(zRange, 0.2), np.less(zRange, 1.0))].mean()
        makeMassLimitVRedshiftPlot(massLimit_90Complete, zRange, diagnosticsDir+os.path.sep+"completeness90Percent#%s.pdf" % (extName), 
                                   title = "%s: $M_{\\rm 500c}$ / $10^{14}$ M$_{\odot}$ > %.2f (0.2 < $z$ < 1)" % (extName, averageMassLimit_90Complete)) 

        # Another potentially useful sanity check plot - the fits themselves
        #plt.ion()
        #plt.close()
        #for key in completenessTab.keys():
            #if key != 'log10M':
                #plt.plot(completenessTab['log10M'], completenessTab[key], 'k.')
                #fitMask=np.equal(fitTab['z'], float(key))
                #plt.plot(completenessTab['log10M'], rayleighFlipped(completenessTab['log10M'], 
                                                                    #fitTab['loc'][fitMask], fitTab['scale'][fitMask]), label = key)
        #plt.legend()

    return fitTab
        
#------------------------------------------------------------------------------------------------------------
def makeMassLimitMap(SNRCut, z, extName, photFilterLabel, mockSurvey, scalingRelationDict, tckQFitDict, 
                     diagnosticsDir):
    """Makes a map of 90% mass completeness (for now, this fraction is fixed).
    
    """
    
    # Get the stuff we need...
    RMSImg=pyfits.open(diagnosticsDir+os.path.sep+"RMSMap_%s#%s.fits" % (photFilterLabel, extName))
    RMSMap=RMSImg[0].data
    areaImg=pyfits.open(diagnosticsDir+os.path.sep+"areaMask#%s.fits" % (extName))
    areaMap=areaImg[0].data
    wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
    RMSTab=getRMSTab(extName, photFilterLabel, diagnosticsDir)

    # Fill in blocks in map for each RMS value, and store giant fit table for this z also
    # Quite possibly we can speed this up loads by doing a subset and interpolating
    outFileName=diagnosticsDir+os.path.sep+"massLimitMap_z%s#%s.fits" % (str(z).replace(".", "p"), extName)
    if os.path.exists(outFileName) == False:
        massLimMap=np.zeros(RMSMap.shape)
        mapFitTab=None
        count=0
        t0=time.time()
        for y0Noise in RMSTab['y0RMS']:
            count=count+1
            print(("... %d/%d (%.3e) ..." % (count, len(RMSTab), y0Noise)))
            fitTab=calcCompleteness(y0Noise, SNRCut, extName, mockSurvey, scalingRelationDict, tckQFitDict, diagnosticsDir,
                                    zRange = [z], fitTabFileName = None)
            fitTab.rename_column('z', 'y0RMS')
            fitTab['y0RMS'][0]=y0Noise
            massLimMap[np.where(RMSMap == y0Noise)]=fitTab['log10MLimit_90%'][0]
            if type(mapFitTab) == type(None):
                mapFitTab=fitTab
            else:
                mapFitTab.add_row(fitTab[0])
        t1=time.time()        
        massLimMap=np.power(10, massLimMap)/1e14
        astImages.saveFITS(outFileName, massLimMap, wcs)

        tabFileName=diagnosticsDir+os.path.sep+"selFn_mapFitTab_z%s#%s.fits" % (str(z).replace(".", "p"), extName)
        if os.path.exists(tabFileName) == True:
            os.remove(tabFileName)
        mapFitTab.write(tabFileName)
    
    # This is sanity checking survey average completeness... 
    # Summary: agrees within 0.2% on average, but some tiles out by up to 3%
    #massImg=pyfits.open(outFileName)
    #massLimMap=massImg[0].data
    #areaImg=pyfits.open(diagnosticsDir+os.path.sep+"areaMask#%s.fits" % (extName))
    #areaMap=areaImg[0].data
    #wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
    #areaMapSqDeg=(mapTools.getPixelAreaArcmin2Map(areaMap, wcs)*areaMap)/(60**2)
    #fromMassLimMap=np.average(massLimMap, weights = areaMapSqDeg/areaMapSqDeg.sum())
    #y0Noise=calcTileWeightedAverageNoise(extName, photFilterLabel, diagnosticsDir)
    #fitTab=calcCompleteness(y0Noise, SNRCut, extName, mockSurvey, parDict['massOptions'], tckQFitDict, diagnosticsDir,
                            #fitTabFileName = diagnosticsDir+os.path.sep+"selFn_fitTab#%s.fits" % (extName))
    #fromSurveyAverage=np.power(10, fitTab['log10MLimit_90%'][np.where(fitTab['z'] == 0.5)])/1e14
    #print("... extName = %s: limit from mass limit map / survey average = %.3f ..." % (extName, fromMassLimMap/fromSurveyAverage))

#------------------------------------------------------------------------------------------------------------
def makeMassLimitVRedshiftPlot(massLimit_90Complete, zRange, outFileName, title = None):
    """Write a plot of 90%-completeness mass limit versus z, adding a spline interpolation.
    
    """
    
    plotSettings.update_rcParams()
    plt.figure(figsize=(9,6.5))
    if title == None:
        ax=plt.axes([0.10, 0.11, 0.87, 0.86])
    else:
        ax=plt.axes([0.10, 0.11, 0.87, 0.80])
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
    if title != None:
        plt.title(title)
    plt.savefig(outFileName)
    plt.close()   
    
#------------------------------------------------------------------------------------------------------------
def cumulativeAreaMassLimitPlot(z, diagnosticsDir):
    """Make a cumulative plot of 90%-completeness mass limit versus survey area, at the given redshift.
    
    """

    fileList=glob.glob(diagnosticsDir+os.path.sep+"massLimitMap_z%s#*.fits" % (str(z).replace(".", "p")))
    extNames=[]
    for f in fileList:
        extNames.append(f.split("#")[-1].split(".fits")[0])
    extNames.sort()
    
    # NOTE: We can avoid this if we add 'areaDeg2' column to selFn_mapFitTab_*.fits tables
    allLimits=[]
    allAreas=[]
    for extName in extNames:
        massLimImg=pyfits.open(diagnosticsDir+os.path.sep+"massLimitMap_z%s#%s.fits" % (str(z).replace(".", "p"), extName))
        massLimMap=massLimImg[0].data
        areaImg=pyfits.open(diagnosticsDir+os.path.sep+"areaMask#%s.fits" % (extName))
        areaMap=areaImg[0].data
        wcs=astWCS.WCS(areaImg[0].header, mode = 'pyfits')
        areaMapSqDeg=(mapTools.getPixelAreaArcmin2Map(areaMap, wcs)*areaMap)/(60**2)
        limits=np.unique(massLimMap).tolist()
        areas=[]
        for l in limits:
            areas.append(areaMapSqDeg[np.where(massLimMap == l)].sum())
        allLimits=allLimits+limits
        allAreas=allAreas+areas
    tab=atpy.Table()
    tab.add_column(atpy.Column(allLimits, 'MLim'))
    tab.add_column(atpy.Column(allAreas, 'areaDeg2'))
    tab.sort('MLim')

    plotSettings.update_rcParams()

    # Full survey plot
    plt.figure(figsize=(9,6.5))
    ax=plt.axes([0.155, 0.12, 0.82, 0.86])
    plt.minorticks_on()
    plt.plot(tab['MLim'], np.cumsum(tab['areaDeg2']), 'k-')
    plt.ylabel("survey area < $M_{\\rm 500c}$ limit (deg$^2$)")
    plt.xlabel("$M_{\\rm 500c}$ (10$^{14}$ M$_{\odot}$) [90% complete]")
    labelStr="total survey area = %.0f deg$^2$" % (np.cumsum(tab['areaDeg2']).max())
    plt.ylim(0.1, 1.01*np.cumsum(tab['areaDeg2']).max())
    plt.xlim(1.5, 10)
    plt.figtext(0.2, 0.9, labelStr, ha="left", va="center")
    plt.savefig(diagnosticsDir+os.path.sep+"cumulativeArea_massLimit_z%s.pdf" % (str(z).replace(".", "p")))
    plt.close()
    
    # Deepest 20%
    totalAreaDeg2=tab['areaDeg2'].sum()
    deepTab=tab[np.where(np.cumsum(tab['areaDeg2']) < 0.2 * totalAreaDeg2)]
    plt.figure(figsize=(9,6.5))
    ax=plt.axes([0.155, 0.12, 0.82, 0.86])
    plt.minorticks_on()
    plt.plot(deepTab['MLim'], np.cumsum(deepTab['areaDeg2']), 'k-')
    plt.ylabel("survey area < $M_{\\rm 500c}$ limit (deg$^2$)")
    plt.xlabel("$M_{\\rm 500c}$ (10$^{14}$ M$_{\odot}$) [90% complete]")
    labelStr="area of deepest 20%% = %.0f deg$^2$" % (np.cumsum(deepTab['areaDeg2']).max())
    plt.ylim(0.1, 1.01*np.cumsum(deepTab['areaDeg2']).max())
    plt.xlim(1.5, deepTab['MLim'].max())
    plt.figtext(0.2, 0.9, labelStr, ha="left", va="center")
    plt.savefig(diagnosticsDir+os.path.sep+"cumulativeArea_massLimit_z%s_deepest20Percent.pdf" % (str(z).replace(".", "p")))
    plt.close()
        
#------------------------------------------------------------------------------------------------------------
def makeFullSurveyMassLimitMapPlot(z, diagnosticsDir):
    """Reprojects tile mass limit maps onto the full map pixelisation, then makes a plot and saves a
    .fits image.
    
    NOTE: we hardcoded the pixelisation in here for now...
    
    """

    # Making the full res reprojected map takes ~1650 sec
    outFileName=diagnosticsDir+os.path.sep+"reproj_massLimitMap_z%s.fits" % (str(z).replace(".", "p"))
    if os.path.exists(outFileName) == False:

        print(">>> Making reprojected full survey mass limit map:")
        
        # Downsampled
        h=pyfits.Header()
        h['NAXIS']=2
        h['NAXIS1']=10800
        h['NAXIS2']=2580
        h['WCSAXES']=2
        h['CRPIX1']=5400.25
        h['CRPIX2']=1890.25
        h['CDELT1']=-0.0333333333333332
        h['CDELT2']=0.0333333333333332
        h['CUNIT1']='deg'
        h['CUNIT2']='deg'
        h['CTYPE1']='RA---CAR'
        h['CTYPE2']='DEC--CAR'
        h['CRVAL1']=0.0
        h['CRVAL2']=0.0
        h['LONPOLE']=0.0
        h['LATPOLE']=90.0
        h['RADESYS']='ICRS'
        # Full res
        #h['NAXIS']=2
        #h['NAXIS1']=43200
        #h['NAXIS2']=10320
        #h['WCSAXES']=2
        #h['CRPIX1']=21601.0
        #h['CRPIX2']=7561.0
        #h['CDELT1']=-0.0083333333333333
        #h['CDELT2']=0.0083333333333333
        #h['CUNIT1']='deg'
        #h['CUNIT2']='deg'
        #h['CTYPE1']='RA---CAR'
        #h['CTYPE2']='DEC--CAR'
        #h['CRVAL1']=0.0
        #h['CRVAL2']=0.0
        #h['LONPOLE']=0.0
        #h['LATPOLE']=90.0
        #h['RADESYS']='ICRS'
        wcs=astWCS.WCS(h, mode = 'pyfits')
        
        reproj=np.zeros([wcs.header['NAXIS2'], wcs.header['NAXIS1']])
        sumPix=np.zeros([wcs.header['NAXIS2'], wcs.header['NAXIS1']])
        
        fileList=glob.glob(diagnosticsDir+os.path.sep+"massLimitMap_z%s#*.fits" % (str(z).replace(".", "p")))
        extNames=[]
        for f in fileList:
            extNames.append(f.split("#")[-1].split(".fits")[0])
        extNames.sort()
        
        t0=time.time()
        for extName in extNames:
            print("... %s ..." % (extName))
            img=pyfits.open(diagnosticsDir+os.path.sep+"massLimitMap_z%s#%s.fits" % (str(z).replace(".", "p"), extName))
            data=img[0].data
            imgWCS=astWCS.WCS(img[0].header, mode = 'pyfits')
            areaImg=pyfits.open(diagnosticsDir+os.path.sep+"areaMask#%s.fits" % (extName))
            areaMap=areaImg[0].data
            for y in range(data.shape[0]):
                for x in range(data.shape[1]):
                    outRADeg, outDecDeg=imgWCS.pix2wcs(x, y)
                    inX, inY=wcs.wcs2pix(outRADeg, outDecDeg)
                    # Once, this returned infinity...
                    try:
                        inX=int(round(inX))
                        inY=int(round(inY))
                    except:
                        continue
                    # handle the overlap regions, which were zeroed
                    if areaMap[y, x] > 0:   
                        reproj[inY, inX]=reproj[inY, inX]+data[y, x]
                        sumPix[inY, inX]=sumPix[inY, inX]+1.0
        t1=time.time()
        #print(t1-t0)
        astImages.saveFITS(outFileName, reproj/sumPix, wcs)
    
    # Make plot
    import colorcet
    img=pyfits.open(outFileName)
    reproj=np.nan_to_num(img[0].data)
    #reproj=np.ma.masked_where(reproj == np.nan, reproj)
    reproj=np.ma.masked_where(reproj <1e-6, reproj)
    wcs=astWCS.WCS(img[0].header, mode = 'pyfits')
    plotSettings.update_rcParams()
    fontSize=20.0
    figSize=(16, 5.7)
    axesLabels="sexagesimal"
    axes=[0.08,0.15,0.91,0.88]
    cutLevels=[2, 9]
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
    plt.figtext(0.53, 0.04, cbLabel, size = 20, ha="center", va="center", fontsize = fontSize, family = "sans-serif")
    plt.savefig(outFileName.replace(".fits", ".pdf"))
    plt.close()
