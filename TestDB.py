# File to load and put data object back on disk
# Adapted from original TestDB which had everything
# static on the windows drive. Now will take a base
# directory to instantiate

import os
import numpy as np
from matplotlib.dates import strpdate2num

class FileDB(object):
    """description of class"""

    def haveData(self, tag, label):
        return os.path.exists(self.getFileName(tag, label))

    def getDirForTag(self, tag):
        return os.path.join(self.baseDir, tag)

    def checkTag(self, tag):
        return os.path.exists(self.getDirForTag(tag))

    def makeTag(self, tag):
        return os.makedirs(self.getDirForTag(tag))

    def getFileName(self, tag, label):
        if (not self.checkTag(tag)): self.makeTag(tag)
        fname = os.path.join(self.getDirForTag(tag), label)
        return fname

    def writeArrayBin(self, tag, label, array):
        fname = self.getFileName(tag, label)
        np.save(fname, array)

    def readArrayBin(self, tag, label):
        try:
            fname = self.getFileName(tag, label)
            arr   = np.load(fname)
            return arr
        except IOError:
            return np.zeros([0,0])
        except TypeError:
            return np.zeros([0,0])

    def writeArray(self, tag, label, array):
        fname = self.getFileName(tag, label)
        np.savetxt(fname, array, '%.6e', delimiter='\t')

    def writeTime(self, tag, label, array):
        fname = self.getFileName(tag, label)
        np.savetxt(fname, array, '%r', delimiter='\t')

    def readArray(self, tag, label):
        try:
            fname   = self.getFileName(tag, label)
            arr     = np.loadtxt(fname)
            return arr
        except IOError:
            return np.zeros([0,0])
        except TypeError:
            return np.zeros([0,0])

    @staticmethod
    def ctonpdt64(x): return np.datetime64(x.decode('utf-8').strip().split("'")[1], 'm')

    def readTime(self, tag, label):
        try:
            fname   = self.getFileName(tag, label)
            arr     = np.loadtxt(fname, converters={0:FileDB.ctonpdt64}, dtype='datetime64')
            return arr
        except IOError:
            return np.zeros([0,0])
        except TypeError:
            return np.zeros([0,0])

    def getLabelsForTag(self, tag):
        # Return all the available labels for a tag
        if (not self.checkTag(tag)): return []
        direc   = self.getDirForTag(tag)
        files   = os.listdir(direc)
        return files

    def __init__(self, basedir):
        self.baseDir = basedir


