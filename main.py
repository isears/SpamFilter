"""
Demo the spam agent that considers single words as independent features of a given document.
Steps:
    * Read in training/testing files in the emails directory
    * Instantiate a spam agent with training files and optimal coefficients (computed using trials on held-out data)
    * Record and print the agent's performance on testing files

"""
from __future__ import division
from spamAgent import *
import os
import time


def main():
    startTime = time.time()
    #Paths to training/testing files
    pathToHamTraining = 'emails/hamtraining/'
    pathToSpamTraining = 'emails/spamtraining/'
    pathToHamTesting = 'emails/hamtesting/'
    pathToSpamTesting = 'emails/spamtesting/'

    hamTrainingFiles = list()
    spamTrainingFiles = list()

    print "reading in ham training files..."
    #construct a list of ham training files
    for filename in os.listdir(pathToHamTraining):
        filePath = pathToHamTraining + filename
        hamTrainingFiles.append(open(filePath, 'r'))

    print "reading in spam training files..."
    #construct a list of spam training files
    for filename in os.listdir(pathToSpamTraining):
        filePath = pathToSpamTraining + filename
        spamTrainingFiles.append(open(filePath, 'r'))

    #Instantiate a spam agent with the spamTraining and hamTraining and train it
    #The coefficients 2 and 1 for features coefficient and Laplacian smoothing coefficient were found by trials on
    #Held-out data
    print "building spam agent..."
    spamFilter = spamClassificationAgent(spamTrainingFiles, hamTrainingFiles, 2, 1)
    print "computing features..."
    spamFilter.computeFeatures()
    print "training spam agent..."
    spamFilter.train()

    spamTotal = 0
    hamTotal = 0
    spamCorrect = 0
    hamCorrect = 0

    print "classifying..."
    #classification tests
    for filename in os.listdir(pathToHamTesting):
        hamTotal += 1
        isSpam = spamFilter.classify(open(pathToHamTesting + filename, 'r'))

        if not isSpam:
            hamCorrect += 1

        print "Classifying Ham file " + filename + ": (should be false)", isSpam

    for filename in os.listdir(pathToSpamTesting):
        spamTotal += 1
        isSpam = spamFilter.classify(open(pathToSpamTesting + filename, 'r'))

        if isSpam:
            spamCorrect += 1

        print "Classifying Spam file " + filename + ": (should be true)", isSpam

    print ""
    print "------------------------------------------------------"
    print "proportion correctly predicted ham: " + str(hamCorrect/hamTotal)
    print "proportion correctly predicted spam: " + str(spamCorrect/spamTotal)
    print "averge proportion correctly predicted: " + str(((hamCorrect/hamTotal)+(spamCorrect/spamTotal))*0.5)
    print "------------------------------------------------------"
    print "execution:", time.time() - startTime, "seconds"

if __name__ == "__main__":
    main()