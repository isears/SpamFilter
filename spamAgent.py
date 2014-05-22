"""
The spam classification agent:

Build a naive bayes model of messages using invididual words as features and spam/ham as classes.
To properly use the spam agent, call the functions in the following order:
    1. __init__(spam_training_files, ham_training_files, k, m)
    2. computeFeatures()
    3. train()
    4. classify(message)
"""
from __future__ import division
from util import *
import math


class spamClassificationAgent:
    def __init__(self, trainingSpam, trainingHam, featuresCoefficient, smoothingCoefficient):
        """
        Spam agent constructor

        \param trainingSpam: A list of spam training emails in the form of opened files
        \param trainingHam: A list of ham training emails in the form of opened files
        \param featuresCoefficient K (0 < int): The number of times a word must appear in the data before it is
            considered as a feature in the naive bayes model. Higher K means the agent will be faster and take up less
            memory, but will probably be less accurate. Lower K can take up more memory/time, but be more accurate. This
            parameter should be adjusted through successive trials of the spam agent on held-out data.
        \param smoothingCoefficient M (0 < int): Laplacian smoothing coefficient. Higher Ms for noisy data, lower Ms
            for more uniform data. This parameter should be adjusted through successive trials of the spam agent on
            held-out data.
        """
        self.tSpam = trainingSpam
        self.tHam = trainingHam
        self.lexicon = list()
        self.k = featuresCoefficient
        self.m = smoothingCoefficient
        self.NGramLexicon = list()
        self.N = 0
        self.totalSpamMessageCount = 0
        self.totalHamMessageCount = 0
        self.totalSpamWordCount = 0
        self.totalHamWordCount = 0

    def computeFeatures(self):
        """
        Find all words that appear more than k times in the training files passed to the agent in the constructor.
        Populate the spam agent's internal lexicon with words found in files.

        Return a list of word objects, each containing the word and a total count (does not compute spam/ham counts for
        each word).
        """
        words = []  # build empty list
        trainingFiles = self.tSpam + self.tHam
        contents = []

        for f in trainingFiles:  # iterates through files in folder
            contents = contents + f.readlines()  # turns contents in big string

        for line in contents:
            for w in line.split():
                words.append(w)

        wordObjects = []  # empty list of objects
        for j in range(len(words)):
            x = word(words[j], 1)
            for i in range(len(wordObjects)):
                if x.name == wordObjects[i].name:  # checks to see if object already in list
                    wordObjects[i].count += 1  # adds to word count if object not in list
                    break
            else:
                wordObjects.append(x)  # adds word object to list if not already there

        for e in range(len(wordObjects)):
            if wordObjects[e].count < self.k:
                wordObjects[e].name = 'null'

        wordObjects = filter(lambda wrd: wrd.name != 'null', wordObjects)  # filters out all flagged objects for list
        self.lexicon = wordObjects
        return wordObjects

    def train(self):
        """
        ***Must be preceeded by a call to computeFeatures function, to populate lexicon correctly***
        Train the agent by determining how many times a word appears in the ham training files and how many times
        it appears in spam training files.
        Update the lexicon to reflect ham/spam counts.

        Return the completed lexicon (a list of word objects with all data fields filled out).
        """
        #find occurances of words in lexicon in ham data
        for f in self.tHam:
            self.totalHamMessageCount += 1
            f.seek(0)
            for fileWord in f.read().split():
                self.totalHamWordCount += 1
                for lexiconWord in self.lexicon:
                    if lexiconWord.name.lower() == fileWord.lower():
                        lexiconWord.hamCount += 1

        #find occurances of words in lexicon in spam data
        for f in self.tSpam:
            self.totalSpamMessageCount += 1
            f.seek(0)
            for fileWord in f.read().split():
                self.totalSpamWordCount += 1
                for lexiconWord in self.lexicon:
                    if lexiconWord.name.lower() == fileWord.lower():
                        lexiconWord.spamCount += 1

        #Laplace smoothing
        for w in self.lexicon:
            w.count += self.m * len(self.lexicon)
            w.hamCount += self.m
            w.spamCount += self.m

        return self.lexicon

    def classify(self, message):
        """
        Classify a message as spam or ham using the naive bayes model built with the computFeatures() and train()
        functions. The classification is achieved by using a maximum likelihood estimation with the posterior
        probabilities for all the words found in the message.

        Procedure:
            1. Initiate hamPartialProbability and spamPartialProbability to 0
            2. For every word in the message that is also in the spam agent's lexicon:
                a. Find the probability that the word appears in the message given it is ham
                b. Take the log base 10 of that probability
                c. Add the resultant value to the hamPartialProbability
                d. Find the probability that the word appears in the message given it is spam
                e. Take the log base 10 of that probability
                f. Add the resultant value to the spamPartialProbability
            3. Take the log base 10 of the ratio of ham messages to spam messages
            4. Add that value to the hamPartialProbability
            5. Take the log base 10 of the ratio of spam messages to ham messages
            6. Add that value to the spamPartialProbability
            7. Compare hamPartialProbability and spamPartialProbability
            8. Return true of the spamPartialProbability is larger, otherwise return false

        Note that summing logarithms of probabilities is used instead of multiplying probabilities to prevent floating
        point underflow.

        \param message: An opened text file to be classified as a spam message or a ham message.

        Return true if message is predicted spam, false if message is predicted ham
        """
        hamPartialProbability = 0
        spamPartialProbability = 0

        for fileWord in message.read().split():
            for lexiconWord in self.lexicon:
                if fileWord.lower() == lexiconWord.name.lower():
                    hamPartialProbability += math.log(lexiconWord.probabilityGivenHam(self.totalHamWordCount))
                    spamPartialProbability += math.log(lexiconWord.probabilityGivenSpam(self.totalSpamWordCount))

        hamPartialProbability += \
            math.log(self.totalHamMessageCount/(self.totalHamMessageCount + self.totalSpamMessageCount))
        spamPartialProbability += \
            math.log(self.totalSpamMessageCount/(self.totalHamMessageCount + self.totalSpamMessageCount))

        if hamPartialProbability < spamPartialProbability:
            ret = True
        else:
            ret = False

        return ret