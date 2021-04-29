# functions that implement analysis and synthesis of sounds using the Harmonic Model
# (for example usage check the models_interface directory)

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import ifft
import math
import dftModel as DFT
import utilFunctions as UF
import sineModel as SM


def f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et):
    """
    Fundamental frequency detection of a sound using twm algorithm
    x: input sound; fs: sampling rate; w: analysis window; 
    N: FFT size; t: threshold in negative dB, 
    minf0: minimum f0 frequency in Hz, maxf0: maximim f0 frequency in Hz, 
    f0et: error threshold in the f0 detection (ex: 5),
    returns f0: fundamental frequency
    """
    if (minf0 < 0):                                            # raise exception if minf0 is smaller than 0
        raise ValueError(
            "Minumum fundamental frequency (minf0) smaller than 0")

    # raise exception if maxf0 is bigger than fs/2
    if (maxf0 >= 10000):
        raise ValueError(
            "Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if (H <= 0):                                               # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")

    # size of positive spectrum
    hN = N//2
    # half analysis window size by rounding
    hM1 = int(math.floor((w.size+1)/2))
    # half analysis window size by floor
    hM2 = int(math.floor(w.size/2))
    # add zeros at beginning to center first window at sample 0
    x = np.append(np.zeros(hM2), x)
    # add zeros at the end to analyze last sample
    x = np.append(x, np.zeros(hM1))
    # init sound pointer in middle of anal window
    pin = hM1
    # last sample to start a frame
    pend = x.size - hM1
    # initialize buffer for FFT
    fftbuffer = np.zeros(N)
    # normalize analysis window
    w = w / sum(w)
    f0 = []                                                    # initialize f0 output
    f0t = 0                                                    # initialize f0 track
    f0stable = 0                                               # initialize f0 stable
    while pin < pend:
        x1 = x[pin-hM1:pin+hM2]                                  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                           # compute dft
        # detect peak locations
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(
            mX, pX, ploc)      # refine peak values
        # convert locations to Hez
        ipfreq = fs * iploc/N
        f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable-f0t) < f0stable/5.0)):
            # consider a stable f0 if it is close to the previous one
            f0stable = f0t
        else:
            f0stable = 0
        # add f0 to output array
        f0 = np.append(f0, f0t)
        pin += H                                                 # advance sound pointer
    return f0


def harmonicDetection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
    """
    Detection of the harmonics of a frame from a set of spectral peaks using f0
    to the ideal harmonic series built on top of a fundamental frequency
    pfreq, pmag, pphase: peak frequencies, magnitudes and phases
    f0: fundamental frequency, nH: number of harmonics,
    hfreqp: harmonic frequencies of previous frame,
    fs: sampling rate; harmDevSlope: slope of change of the deviation allowed to perfect harmonic
    returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
    """

    if (f0 <= 0):                                          # if no f0 return no harmonics
        return np.zeros(nH), np.zeros(nH), np.zeros(nH)
    # initialize harmonic frequencies
    hfreq = np.zeros(nH)
    # initialize harmonic magnitudes
    hmag = np.zeros(nH)-100
    # initialize harmonic phases
    hphase = np.zeros(nH)
    # initialize harmonic frequencies
    hf = f0*np.arange(1, nH+1)
    hi = 0                                               # initialize harmonic index
    # if no incomming harmonic tracks initialize to harmonic series
    if hfreqp == []:
        hfreqp = hf
    # find harmonic peaks
    while (f0 > 0) and (hi < nH) and (hf[hi] < fs/2):
        pei = np.argmin(abs(pfreq - hf[hi]))               # closest peak
        # deviation from perfect harmonic
        dev1 = abs(pfreq[pei] - hf[hi])
        dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi] >
                0 else fs)  # deviation from previous frame
        threshold = f0/3 + harmDevSlope * pfreq[pei]
        if ((dev1 < threshold) or (dev2 < threshold)):         # accept peak if deviation is small
            # harmonic frequencies
            hfreq[hi] = pfreq[pei]
            # harmonic magnitudes
            hmag[hi] = pmag[pei]
            hphase[hi] = pphase[pei]                         # harmonic phases
        hi += 1                                            # increase harmonic index
    return hfreq, hmag, hphase


def harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et):
    """
    Analysis/synthesis of a sound using the sinusoidal harmonic model
    x: input sound, fs: sampling rate, w: analysis window, 
    N: FFT size (minimum 512), t: threshold in negative dB, 
    nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
    maxf0: maximim f0 frequency in Hz, 
    f0et: error threshold in the f0 detection (ex: 5),
    returns y: output array sound
    """

    hN = N//2                                               # size of positive spectrum
    # half analysis window size by rounding
    hM1 = int(math.floor((w.size+1)/2))
    # half analysis window size by floor
    hM2 = int(math.floor(w.size/2))
    # add zeros at beginning to center first window at sample 0
    x = np.append(np.zeros(hM2), x)
    # add zeros at the end to analyze last sample
    x = np.append(x, np.zeros(hM1))
    # FFT size for synthesis (even)
    Ns = 512
    # Hop size used for analysis and synthesis
    H = Ns/4
    hNs = Ns/2
    # init sound pointer in middle of anal window
    pin = max(hNs, hM1)
    # last sample to start a frame
    pend = x.size - max(hNs, hM1)
    # initialize buffer for FFT
    fftbuffer = np.zeros(N)
    # initialize output sound frame
    yh = np.zeros(Ns)
    # initialize output array
    y = np.zeros(x.size)
    # normalize analysis window
    w = w / sum(w)
    # initialize synthesis window
    sw = np.zeros(Ns)
    # overlapping window
    ow = triang(2*H)
    sw[hNs-H:hNs+H] = ow
    bh = blackmanharris(Ns)                                 # synthesis window
    # normalize synthesis window
    bh = bh / sum(bh)
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / \
        bh[hNs-H:hNs+H]     # window for overlap-add
    hfreqp = []
    f0t = 0
    f0stable = 0
    while pin < pend:
        # -----analysis-----
        x1 = x[pin-hM1:pin+hM2]                               # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
        # detect peak locations
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(
            mX, pX, ploc)   # refine peak values
        ipfreq = fs * iploc/N
        f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable-f0t) < f0stable/5.0)):
            # consider a stable f0 if it is close to the previous one
            f0stable = f0t
        else:
            f0stable = 0
        hfreq, hmag, hphase = harmonicDetection(
            ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs)  # find harmonics
        hfreqp = hfreq
    # -----synthesis-----
        Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns,
                             fs)     # generate spec sines
        fftbuffer = np.real(ifft(Yh))                         # inverse FFT
        # undo zero-phase window
        yh[:hNs-1] = fftbuffer[hNs+1:]
        yh[hNs-1:] = fftbuffer[:hNs+1]
        y[pin-hNs:pin+hNs] += sw*yh                           # overlap-add
        pin += H                                              # advance sound pointer
    # delete half of first window which was added in stftAnal
    y = np.delete(y, range(hM2))
    # add zeros at the end to analyze last sample
    y = np.delete(y, range(y.size-hM1, y.size))
    return y


def harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope=0.01, minSineDur=.02):
    """
    Analysis of a sound using the sinusoidal harmonic model
    x: input sound; fs: sampling rate, w: analysis window; N: FFT size (minimum 512); t: threshold in negative dB, 
    nH: maximum number of harmonics;  minf0: minimum f0 frequency in Hz, 
    maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
    harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
    returns xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases
    """

    # raise exception if minSineDur is smaller than 0
    if (minSineDur < 0):
        raise ValueError("Minimum duration of sine tracks smaller than 0")

    hN = N//2                                               # size of positive spectrum
    # half analysis window size by rounding
    hM1 = int(math.floor((w.size+1)/2))
    # half analysis window size by floor
    hM2 = int(math.floor(w.size/2))
    # add zeros at beginning to center first window at sample 0
    x = np.append(np.zeros(hM2), x)
    # add zeros at the end to analyze last sample
    x = np.append(x, np.zeros(hM2))
    # init sound pointer in middle of anal window
    pin = hM1
    # last sample to start a frame
    pend = x.size - hM1
    # initialize buffer for FFT
    fftbuffer = np.zeros(N)
    # normalize analysis window
    w = w / sum(w)
    # initialize harmonic frequencies of previous frame
    hfreqp = []
    f0t = 0                                                 # initialize f0 track
    f0stable = 0                                            # initialize f0 stable
    while pin <= pend:
        x1 = x[pin-hM1:pin+hM2]                               # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
        # detect peak locations
        ploc = UF.peakDetection(mX, t)
        iploc, ipmag, ipphase = UF.peakInterp(
            mX, pX, ploc)   # refine peak values
        ipfreq = fs * iploc/N                                 # convert locations to Hz
        f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable-f0t) < f0stable/5.0)):
            # consider a stable f0 if it is close to the previous one
            f0stable = f0t
        else:
            f0stable = 0
        hfreq, hmag, hphase = harmonicDetection(
            ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope)  # find harmonics
        hfreqp = hfreq
        if pin == hM1:                                        # first frame
            xhfreq = np.array([hfreq])
            xhmag = np.array([hmag])
            xhphase = np.array([hphase])
        else:                                                 # next frames
            xhfreq = np.vstack((xhfreq, np.array([hfreq])))
            xhmag = np.vstack((xhmag, np.array([hmag])))
            xhphase = np.vstack((xhphase, np.array([hphase])))
        pin += H                                              # advance sound pointer
    # delete tracks shorter than minSineDur
    xhfreq = SM.cleaningSineTracks(xhfreq, round(fs*minSineDur/H))
    return xhfreq, xhmag, xhphase
