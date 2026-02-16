# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt

plt.rcParams['svg.fonttype'] = 'none'
import os
import pickle
import cv2
from tqdm import tqdm
from scipy.signal import butter, freqz, filtfilt, firwin, bode
from IPython.display import clear_output
import os.path
from os import path
from matplotlib import pyplot as plt
import csv
from scipy import stats
from src2.shared.path_finder import PathFinder


def _prepAxes(title='', xLabel='', yLabel='', subPlots=None):
    """
    Prepare figure and axis/axes for plotting. Returns the figure handle and either the axis handle or a list of axes handles.
    TITLE is the title of the axis or list of titles (in order) for the axes.
    XLABEL is the xlabel of the axis or list of xlabels (in order) for the axes.
    YLABEL is the ylabel of the axis or list of ylabels (in order) for the axes.
    SUBPLOTS is a list to prepare the subplot axes: [number of rows, number of columns].
    The elements in TITLE, XLABEL, and YLABEL label the plots first from left to right, then from top to bottom.
    """
    if isinstance(xLabel, list) and isinstance(yLabel, list):
        if len(xLabel) > len(yLabel):
            while len(xLabel) is not len(yLabel):
                yLabel.append('')
        if len(yLabel) > len(xLabel):
            while len(xLabel) is not len(yLabel):
                xLabel.append('')
    elif isinstance(xLabel, list) and isinstance(yLabel, str):
        yLabel.append('')
        while len(xLabel) is not len(yLabel):
            yLabel.append('')
    elif isinstance(xLabel, str) and isinstance(yLabel, list):
        xLabel.append('')
        while len(xLabel) is not len(yLabel):
            xLabel.append('')

    h = plt.figure()
    # h.set_layout_engine('constrained')
    if subPlots is None:
        ax = h.add_subplot()
        ax.set_title(title)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
    else:
        ax = []
        if type(title) is str:
            title = [title] * (subPlots[0] * subPlots[1])
        if type(xLabel) is str:
            xLabel = [xLabel] * (subPlots[0] * subPlots[1])
        if type(yLabel) is str:
            yLabel = [yLabel] * (subPlots[0] * subPlots[1])
        for k in range(subPlots[0] * subPlots[1]):
            ax.append(h.add_subplot(subPlots[0], subPlots[1], k + 1))
            if k <= len(title):
                ax[k].set_title(title[k])
            if k <= len(xLabel):
                ax[k].set_xlabel(xLabel[k])
            if k <= len(yLabel):
                ax[k].set_ylabel(yLabel[k])
    h.tight_layout() # incompatible with the 'constrained' layout engine
    return h, ax


def spectrogram(tVec, freqVec, specData, cBarPercentLims=[5., 95.], xLabel='Time (s)', yLabel='Frequency (Hz)',
                cLabel='Power (dB)'):
    """
    Plots a spectrogram that has already been computed.
    TVEC is a vector of the x-axis time points or a time vector consisting of just [min, max].
    FREQVEC is a vector of the y-axis frequency points, or a frequency vector consisting of just [min, max].
    SPECDATA is the matrix of spectral power.
    CBARPERCENTLIMS sets the bounds on the color bar by finding the specified percentages of the power in specData.
    """
    h, ax = _prepAxes(xLabel=xLabel, yLabel=yLabel)
    cBarMin = np.percentile(specData, cBarPercentLims[0])
    cBarMax = np.percentile(specData, cBarPercentLims[1])
    spectrogramPlot = ax.imshow(specData, interpolation='none', extent=[tVec[0], tVec[-1], freqVec[0], freqVec[-1]],
                                aspect='auto', vmin=cBarMin, vmax=cBarMax, origin='lower')
    cbar = h.colorbar(spectrogramPlot, ax=ax)
    cbar.set_label(cLabel)
    return h, ax


def markEvents(axisHandle, eventTimes):
    ##Mark Neuralynx events on a given plot
    yLimits = axisHandle.get_ylim()
    xLimits = axisHandle.get_xlim()
    lineLength = np.diff(yLimits)
    lineOffset = yLimits[0] + (lineLength / 2)
    if type(eventTimes) != list:
        eventTimes = [eventTimes]
    axisHandle.eventplot(eventTimes, lineoffsets=lineOffset, linelengths=lineLength, colors='k')
    axisHandle.axis([xLimits[0], xLimits[1], yLimits[0], yLimits[1]])


def _findFilePaths(directory=None, fileExtensions=None, fileStartsWith=None,
                   removeFile=False, printPath=False, fileAndDirectory=False):
    '''
    timeStampsFilename = misc_functions._findFilePaths(self.experiment['calcium imaging directory'], '.csv', 'timeStamps', removeFile=False, printPath=False, fileAndDirectory=False)

    
    
    Makes a list of the full paths of all files of type FILEEXTENSIONS in DIRECTORY, sorted by the time they were last modified.
    FILEEXTENSIONS is a string of the file extension or a list or tuple with multiple file extensions.
    FILESTARTSWITH is a string or tuple of strings to be included.
    REMOVEFILE returns path of folder containing the files you want
    '''

    if (fileExtensions == None and fileStartsWith == None):
        raise AttributeError('Not enough information to determine path')

    if printPath:
        print('Finding file path...')
        print('directory=' + str(directory))
        print('fileExtensions=' + str(fileExtensions))
        print('fileStartsWith=' + str(fileStartsWith))
    if fileExtensions != None:
        if type(fileExtensions) is str:
            fileExtensionsTuple = (fileExtensions,)
        else:
            fileExtensionsTuple = tuple(fileExtensions)
    else:
        fileExtensionsTuple = None
    if fileStartsWith != None:
        if type(fileStartsWith) is str:
            fileStartsWith = (fileStartsWith,)
        else:
            fileStartsWith = tuple(fileStartsWith)
    filePaths = []
    fileDirectory = []

    for root, dirs, files in os.walk(directory):
        for file1 in files:
            if fileExtensionsTuple != None:
                if file1.endswith(fileExtensionsTuple):
                    if fileStartsWith == None:
                        if removeFile:
                            filePaths.append(os.path.join(root))
                        elif fileAndDirectory:
                            filePaths.append(os.path.join(root, file1))
                            fileDirectory.append(os.path.join(root))
                        else:
                            filePaths.append(os.path.join(root, file1))
                    elif file1.startswith(fileStartsWith):
                        if removeFile:
                            filePaths.append(os.path.join(root))
                        elif fileAndDirectory:
                            filePaths.append(os.path.join(root, file1))
                            fileDirectory.append(os.path.join(root))
                        else:
                            filePaths.append(os.path.join(root, file1))
            else:
                if file1.startswith(fileStartsWith):
                    if removeFile:
                        filePaths.append(os.path.join(root))
                    elif fileAndDirectory:
                        filePaths.append(os.path.join(root, file1))
                        fileDirectory.append(os.path.join(root))
                    else:
                        filePaths.append(os.path.join(root, file1))

    if filePaths == []:
        raise AttributeError('No path found')
    if printPath:
        print('filePaths=' + str(filePaths))
    if fileAndDirectory:
        return (sorted(set(filePaths), key=os.path.getmtime), sorted(set(fileDirectory), key=os.path.getmtime))
    else:
        return sorted(set(filePaths), key=os.path.getmtime)


def loadObj(filename):
    ##Loads a pickled object into memory from FILENAME. Useful for loading a previously used instance of a class (e.g., miniscope_Ephys.miniscopeEphys class).##
    fileToRead = open(filename, 'rb')
    loadedObject = pickle.load(fileToRead)
    fileToRead.close()
    return loadedObject


def denoiseMovie(dataDir, dataFilePrefix='', showVideo=False, startingFileNum=0,
                 framesPerFile=1000, fs=30, frameStep=10, goodRadius=2000,
                 notchHalfWidth=3, centerHalfHeightToLeave=90, cutoff=3.0,
                 butterOrder=6, mode='display', compressionCodec='FFV1', jobID=''):
    '''
    Loads a movie and removes both the horizontal bands (that slowly travel upwards) from the movie and the slow flicker of the entire image. Code largely stolen from Daniel Aharoni's python notebook (https://github.com/Aharoni-Lab/Miniscope-v4/tree/master/Miniscope-v4-Denoising-Notebook).
    DATADIR is the directory that contains the movie files to be denoised.
    DATAFILEPREFIX is anything that comes before the number of the movie file. For example, the file 'msCam0.avi' would have a prefix of 'msCam'.
    SHOWVIDEO determines whether to display the movie file prior to beginning the analysis.
    STARTINGFILENUM is the starting number of the sequence of movie files to be denoised. All files with numbers greater than or equal to this will be denoised.
    FRAMESPERFILE is the number of frames in each file. This number is set by the Miniscope software.
    FS is the movie frame acquisition rate.
    FRAMESTEP is the step size for generating the 2D FFT. This can be used to skip frames and speed up processing.
    GOODRADIUS 
    NOTCHHALFWIDTH
    CENTERHALFHEIGHTTOLEAVE is the half-height of the pass frequencies in the 2D FFT.
    CUTOFF
    BUTTERORDER generally between 4-9 or there will be more artifacts
    MODE determines whether to 'save' or 'display' the denoised movie.
    COMPRESSIONCODEC determines the compression codec to use. Options are 'FFV1' or 'GREY'.
    Makes sure path ends with '/'
    '''
    difVideos = []

    if not isinstance(dataDir, list):
        dataDir = [dataDir]
    print(f"This is our dataDir: {dataDir}")
    for filePath in dataDir:
        print(f"This is our filePath: {filePath}")
        if (filePath + '\Denoised') in dataDir:
            print('already denoised')
            print('skip=' + filePath)
            continue

        if 'Denoised' in filePath:
            print('skip=' + filePath)
            continue

        # Makes sure path ends with '/'

        if filePath[-1] != "/":
            filePath = filePath + "/"
        print('filePath=' + filePath)

        # -------------------------------
        # Run through avi files and generate mean fft

        rows = 0
        cols = 0
        # -----------------------

        fileNum = startingFileNum
        sumFFT = None
        applyVignette = True
        vignetteCreated = False
        running = True

        while (path.exists(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum)) and running is True):
            cap = cv2.VideoCapture(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))
            fileNum = fileNum + 1
            frameNum = 0
            for frameNum in tqdm(range(0, framesPerFile, frameStep), total=framesPerFile / frameStep,
                                 desc="Running file {:.0f}.avi".format(fileNum - 1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
                ret, frame = cap.read()

                if (vignetteCreated is False):
                    rows, cols = frame.shape[:2]
                    X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 4)
                    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 4)
                    resultant_kernel = Y_resultant_kernel * X_resultant_kernel.T
                    mask = 255 * resultant_kernel / np.linalg.norm(resultant_kernel)
                    vignetteCreated = True

                if applyVignette is False:
                    mask = 1

                if (ret is False):
                    break
                else:
                    frame = frame[:, :, 1] * mask

                    dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)
                    dft_shift = np.fft.fftshift(dft)

                    try:
                        sumFFT = sumFFT + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
                    except:
                        sumFFT = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

                    if (showVideo is True):
                        cv2.imshow("Vid", frame / 255)
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            running = False
                            break
        cv2.destroyAllWindows()

        # Modify FFT using a circle mask around center
        # -----------------------
        crow, ccol = int(rows / 2), int(cols / 2)

        maskFFT = np.zeros((rows, cols, 2), np.float32)
        cv2.circle(maskFFT, (crow, ccol), goodRadius, 1, thickness=-1)

        # for i in cutFreq:
        #     maskFFT[(i + crow-notchHalfWidth):(i+crow+notchHalfWidth),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
        #     maskFFT[(-i + crow-notchHalfWidth):(-i+crow+notchHalfWidth),(ccol-notchHalfWidth):(ccol+notchHalfWidth),0] = 0
        maskFFT[(crow + centerHalfHeightToLeave):, (ccol - notchHalfWidth):(ccol + notchHalfWidth), 0] = 0
        maskFFT[:(crow - centerHalfHeightToLeave), (ccol - notchHalfWidth):(ccol + notchHalfWidth), 0] = 0

        maskFFT[:, :, 1] = maskFFT[:, :, 0]

        modifiedFFT = sumFFT * maskFFT[:, :, 0]

        """
        # Plot original and modified FFT
        plt.figure()
        plt.subplot(121),plt.imshow(np.log(sumFFT), cmap = 'gray')
        plt.title('Mean FFT of Data')
        plt.subplot(122),plt.imshow(np.log(modifiedFFT), cmap = 'gray')
        plt.title('Filtered FFT')
        """

        # Display filtered vs original videos
        # -----------------------
        if showVideo:
            fileNum = startingFileNum
            sumFFT = None
            running = True

            while (path.exists(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum)) and running is True):
                cap = cv2.VideoCapture(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))
                fileNum = fileNum + 1
                for frameNum in tqdm(range(0, framesPerFile, frameStep), total=framesPerFile / frameStep,
                                     desc="Running file {:.0f}.avi".format(fileNum - 1)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
                    ret, frame = cap.read()

                    if (ret is False):
                        break
                    else:
                        frame = frame[:, :, 1]
                        dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
                        dft_shift = np.fft.fftshift(dft)

                        fshift = dft_shift * maskFFT
                        f_ishift = np.fft.ifftshift(fshift)
                        img_back = cv2.idft(f_ishift)
                        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

                        img_back[img_back > 255] = 255
                        img_back = np.uint8(img_back)

                        im_diff = (128 + (frame - img_back) * 2)
                        im_v = cv2.hconcat([frame, img_back, im_diff])
                        cv2.imshow("Raw, Filtered, Difference", im_v / 255)

                        try:
                            sumFFT = sumFFT + cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
                        except:
                            sumFFT = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            running = False
                            break

            cv2.destroyAllWindows()

        # Calculate mean fluorescence per frame
        # Users shouldn't change anything here
        frameStep = 1  # Should stay as 1
        fileNum = startingFileNum
        sumFFT = None
        meanFrameList = []
        while (path.exists(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))):
            cap = cv2.VideoCapture(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))
            fileNum = fileNum + 1
            for frameNum in tqdm(range(0, framesPerFile, frameStep), total=framesPerFile / frameStep,
                                 desc="Running file {:.0f}.avi".format(fileNum - 1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
                ret, frame = cap.read()
                if (ret is False):
                    break
                else:
                    frame = frame[:, :, 1]
                    dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
                    dft_shift = np.fft.fftshift(dft)

                    fshift = dft_shift * maskFFT
                    f_ishift = np.fft.ifftshift(fshift)
                    img_back = cv2.idft(f_ishift)
                    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
                    meanFrameList.append(img_back.mean())

                    # clear_output(wait=True)

                    # plt.subplot(121),plt.imshow(frame, cmap = 'gray')
                    # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
                    # plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
                    # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])

                    # plt.show()

        meanFrame = np.array(meanFrameList)

        # Create a lowpass filter
        # Sample rate and desired cutoff frequencies (in Hz).

        # -----------------------

        plt.figure()
        for order in [3, 6, 9]:
            b, a = butter(order, cutoff / (0.5 * fs), btype='low', analog=False)
            w, h = freqz(b, a, worN=2000)
            # plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        """
        plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
                     '--', label='sqrt(0.5)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.grid(True)
        plt.legend(loc='best')
        """
        # Plot Mean Frame Resuls
        # plt.figure(figsize=(8,4))
        # plt.plot(meanFrame)

        # Plot effect of filtering

        # -----------------------

        b, a = butter(butterOrder, cutoff / (0.5 * fs), btype='low', analog=False)
        try:
            meanFiltered = filtfilt(b, a, meanFrame)
        except:
            print("ERROR:" + filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))
            difVideos.append(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))
            continue

        '''
        plt.figure()
        plt.plot(meanFrame, 'k', label='Raw Data')
        plt.plot( meanFiltered, label='Filtered Data')
        plt.plot(meanFrame - meanFiltered,'r', label='Difference')
        plt.xlabel('frame number')
        # plt.hlines([-a, a], 0, T, linestyles='--')
        plt.grid(True)
        plt.axis('tight')
        plt.legend(loc='upper left')
        '''
        # meanFrame[3000]

        # Apply FFT spatial filtering and lowpass filtering to data and has the option of saving as new videos

        if mode == 'save':
            frameStep = 1

        # --------------------

        fileNum = startingFileNum
        sumFFT = None
        frameCount = 0
        running = True

        codec = cv2.VideoWriter_fourcc(compressionCodec[0], compressionCodec[1], compressionCodec[2],
                                       compressionCodec[3])

        if mode == "save" and not path.exists(filePath + "Denoised"):
            os.mkdir(filePath + "Denoised")

        while not (not path.exists(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum)) or not (running is True)):
            cap = cv2.VideoCapture(filePath + dataFilePrefix + "{:.0f}.avi".format(fileNum))

            if mode == "save":
                writeFile = cv2.VideoWriter(
                    filePath + "Denoised/" + jobID + dataFilePrefix + "denoised{:.0f}.avi".format(fileNum),
                    codec, 60, (cols, rows), isColor=False)

            fileNum = fileNum + 1
            # frameNum = 0
            for frameNum in tqdm(range(0, framesPerFile, frameStep), total=framesPerFile / frameStep,
                                 desc="Running file {:.0f}.avi".format(fileNum - 1)):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frameNum)
                ret, frame = cap.read()
                # frameNum = frameNum + frameStep 

                # print(frameCount)

                if (ret is False):
                    break
                else:
                    frame = frame[:, :, 1]
                    dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT | cv2.DFT_SCALE)
                    dft_shift = np.fft.fftshift(dft)

                    fshift = dft_shift * maskFFT
                    f_ishift = np.fft.ifftshift(fshift)
                    img_back = cv2.idft(f_ishift)
                    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

                    meanF = img_back.mean()
                    img_back = img_back * (1 + (meanFiltered[frameCount] - meanF) / meanF)
                    img_back[img_back > 255] = 255
                    img_back = np.uint8(img_back)

                    if mode == "save":
                        writeFile.write(img_back)

                    if mode == "display":
                        im_diff = (128 + (frame - img_back) * 2)
                        im_v = cv2.hconcat([frame, img_back])
                        im_v = cv2.hconcat([im_v, im_diff])

                        im_v = cv2.hconcat([frame, img_back, im_diff])
                        cv2.imshow("Cleaned video", im_v / 255)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            running = False
                            cap.release()
                            break

                    frameCount = frameCount + 1

            if mode == "save":
                writeFile.release()

        cv2.destroyAllWindows()
    if len(difVideos) != 0:
        print('ERRORS with: ' + str(difVideos))
        print('Consider investigating')


def importVideoAsNumpyArray(filename, frames='all', displayFrame=False, frameToDisplay=10):
    """Code stolen from https://stackoverflow.com/questions/42163058/how-to-turn-a-video-into-numpy-array and edited."""
    cap = cv2.VideoCapture(filename)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames != 'all':
        frameCount = frames
    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while (fc < frameCount and ret):
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    if displayFrame:
        cv2.namedWindow('frame ' + str(frameToDisplay))
        cv2.imshow('frame ' + str(frameToDisplay), buf[frameToDisplay - 1])
    return buf


def quatToEuler(qw, qx, qy, qz, degrees=False):
    m00 = 1.0 - 2.0 * qy * qy - 2.0 * qz * qz
    m01 = 2.0 * qx * qy + 2.0 * qz * qw
    m02 = 2.0 * qx * qz - 2.0 * qy * qw
    m10 = 2.0 * qx * qy - 2.0 * qz * qw
    m11 = 1 - 2.0 * qx * qx - 2.0 * qz * qz
    m12 = 2.0 * qy * qz + 2.0 * qx * qw
    m20 = 2.0 * qx * qz + 2.0 * qy * qw
    m21 = 2.0 * qy * qz - 2.0 * qx * qw
    m22 = 1.0 - 2.0 * qx * qx - 2.0 * qy * qy

    eulerAngles = []

    R = np.arctan2(m12, m22)  ##Roll
    eulerAngles.append(R)
    c2 = np.sqrt(m00 * m00 + m01 * m01)
    P = np.arctan2(-m02, c2)  ##Pitch
    eulerAngles.append(P)
    s1 = np.sin(R)
    c1 = np.cos(R)
    Y = np.arctan2(s1 * m20 - c1 * m10, c1 * m11 - s1 * m21)  ##Yaw
    eulerAngles.append(Y)
    if degrees == True:
        eulerAngles = [math.degrees(R), math.degrees(P), math.degrees(Y)]
    return eulerAngles


def _convQuatToEuler(line):
    if len(line) != 5:
        print('!!! ERROR: Invalid file')  # FIXME
        return
    time = line[0]
    qw = line[1]
    qx = line[2]
    qy = line[3]
    qz = line[4]
    eulerAngles = list(quatToEuler(qw, qx, qy, qz, degrees=False))
    eulerAngles.insert(0, time)  ##time in matrix?
    return eulerAngles


def _calcNumMinusMean(num, mean):
    return (num - mean)


def _compVThresh(num, VThresh):
    x = int(abs(num) >= abs(VThresh))
    return x


def _findStepIndex(conArray):
    x = np.diff(np.round(np.diff(conArray), 3))
    index = np.asarray(np.where(abs(x) > 1)[0]) + 1
    return index


def threshFunc(dataArray, threshVal):
    dataArray = np.where(dataArray >= threshVal, 1, 0)
    dataArray = np.argwhere(np.diff(dataArray) == 1)
    addArr = np.zeros(np.shape(dataArray))
    addArr[..., -1] = 1
    dataArray = dataArray + addArr
    return dataArray


def filterData(data, n, cut, ftype, btype, fs, bodePlot=False):
    """ Use ftype to indicate FIR or Butterworth filter.
    
    For the FIR filter indicate a LowPass, HighPass, or BandPass with btype = lowpass, highpass, or bandpass, respectively. 
    n is the length of the filter (number of coefficients, i.e. the filter order + 1). numtaps must be odd if a passband includes the Nyquist frequency.
    A good value for n is 10000.
    Channel should be set to desired .ncs file
    
    The Butterworth filters have a more linear phase response in the pass-band than other types and is able to provide better group delay performance, and also a lower level of overshoot.
    Indicate the filter type by setting btype = 'low', 'high', or 'band'.
    The default for n is n = 2
    For a bandpass filter indicate the lowstop and the highstop by using an array. example: wn= ([10, 30])"""

    if ftype.lower() == 'fir':
        h = firwin(n, cut, pass_zero=btype, fs=fs)  # Build the FIR filter
        filteredData = filtfilt(h, 1, data)  # Zero-phase filter the data
        if bodePlot:
            w, a = freqz(h, worN=10000,fs=2000)
            plt.figure()
            plt.semilogx(w, abs(a))
            
            w, mag, phase = bode((h,1),w=2*np.pi*w)
            plt.figure()
            plt.semilogx(w,mag)
            plt.figure()
            plt.semilogx(w,phase)

    if ftype.lower() == 'butterworth' or ftype.lower() == 'butter':
        b, a = butter(n, cut, btype=btype, fs=fs)
        filteredData = filtfilt(b, a, data)
        
        if bodePlot:
            w, h = freqz(b, a, worN=10000,fs=2000)
            plt.figure()
            plt.semilogx(w, abs(h))
            
            w, mag, phase = bode((b,a),w=2*np.pi*w)
            plt.figure()
            plt.semilogx(w,mag)
            plt.figure()
            plt.semilogx(w,phase)

    return filteredData


def updateCSVCell(data, columnTitle, lineNum, csvFile):
    # get the correct column
    with open(csvFile) as file:
        reader = csv.DictReader(file)
        csvData = []
        for row in reader:
            csvData.append(row)
            if dict(row).get('line number') == str(lineNum):
                row[columnTitle] = str(data)

        with open(csvFile, 'w', newline='') as writeFile:
            writer = csv.DictWriter(writeFile, fieldnames=reader.fieldnames)
            writer.writeheader()
            for lineData in csvData:
                writer.writerow(lineData)


def appendRowCSV(data, filename):
    """Appends a new row to a CSV file.
    Args:
        data: Dictionary of data to be added to the csv file
        filename: Name of the CSV file to write to.
    """
    if not os.path.exists(filename):
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, dict(data).keys())
            writer.writeheader()
            writer.writerow(dict(data))
    else:
        with open(filename, 'a', newline='') as file:
            writer = csv.DictWriter(file, dict(data).keys())
            writer.writerow(dict(data))


def spikeTrigAvg(eventArray, dataArray, framesb, framesa):       
    """
    Compute the average spike values starting 'framesb' before the event
    and ending 'framesa' after the event.
    
    Args:
        eventArray: A numpy array of when and/or where events occur. Can either
                    be in the format of [[component, frame],...] or
                    [[frame],...]
        dataArray: A numpy array of the signal values at each frame
        framesb, framesa: the number of frames before and after the event that
                          that are to be included in the average spike
    Returns:
        avgEventDict: a dictionary dictionary where the keys represent the
                      component number from the dataArray and the value is
                      a numpy array of the average values at each frame
                      of the designated window around the event
    """
    avgEventDict = {}
    if dataArray.ndim == 1:
        for event in eventArray:
            if event[0]>=framesb and event[0]<=dataArray.size-framesa-1:
                if 0 in avgEventDict.keys():
                    avgEventDict[0] = np.add(avgEventDict[0], dataArray[int(event[0])-framesb:int(event[0])+framesa+1], dtype=object)
                else:
                    avgEventDict[0] = dataArray[int(event[0])-framesb:int(event[0])+framesa+1]
        avgEventDict[0] /= len(eventArray)
    else:
        for event in eventArray:
            if event[1]>=framesb and event[1]<=dataArray[0].size-framesa-1:
                if event[0] in avgEventDict.keys():
                    avgEventDict[int(event[0])] = np.add(avgEventDict[int(event[0])], dataArray[int(event[0])][int(event[1])-framesb:int(event[1])+framesa+1], dtype=object)
                else:
                    avgEventDict[int(event[0])] = dataArray[int(event[0])][int(event[1])-framesb:int(event[1])+framesa+1]
        for component in range(0,len(dataArray)):
            if component in avgEventDict.keys():
                avgEventDict[component] /= len(np.argwhere(eventArray==component))
    return avgEventDict


def zScore(dataArray, frameWindow = 1000):
    """
    Compute the z-score of the data array values every designated frame window
    length based on the values within that frame window
    
    Args:
        dataArray: A numpy array of values where the row represents the component
                   and the column represents the frame number
        frameWindow: An integer value that determines the length of the window
                     which the function z-scores across. Defaults to 1000 frames
    
    Returns:
        zScoreArray: A numpy array of the same shape as dataArray containing the 
                     z-score values of each frame
    """
    
    zScoreArray = np.ndarray(np.shape(dataArray))
    for i in range(0, int(dataArray[0].size/frameWindow)):
        if i*frameWindow < dataArray[0].size - frameWindow:
            zScoreArray[:][i*frameWindow:i*frameWindow+frameWindow] = stats.zscore(dataArray[:][i*frameWindow:i*frameWindow+frameWindow], axis=1)
            
        else:
            zScoreArray[:][i*frameWindow:] = stats.zscore(dataArray[:][i*frameWindow:], axis=1)
    zScoreArray = np.nan_to_num(zScoreArray)
    return zScoreArray


def get_coords_dict_from_analysis_params(miniscope_data_manager, crop=False, crop_square=False):
    coords_dict = None
    crop_job_name = ''
    try:
        if crop:
            previous_coords = miniscope_data_manager.analysis_params['crop']
            coords_dict = { 'x0': previous_coords[0], 'y0': previous_coords[1], 'x1': previous_coords[2], 'y1': previous_coords[3]}
            crop_job_name = '_crop'
        elif crop_square:
            previous_coords = miniscope_data_manager.analysis_params['crop_square']
            coords_dict = { 'x0': previous_coords[0], 'y0': previous_coords[1], 'x1': previous_coords[2], 'y1': previous_coords[3]}
            crop_job_name = '_crop_square'
    except:
        print("Did not find any coords under ", 'crop' if crop else 'crop_square')
    
    return coords_dict, crop_job_name
        
