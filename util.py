"""
Define utility classes used by the spamAgent
"""


class word:
    """
    Store information about a word object. Word objects make up the spam agent's lexicon.
    """
    def __init__(self, wordName, wordCount):
        """
        Word object constructor

        \param wordName (string): The word as a string
        \param wordCount (int): The number of times the word appears in the training data

        When the word object is first instantiated, spamCount and hamCount are initialized to 0, which means that
        calling the probabilityGivenSpam() or probabilityGivenHam() methods will always return 0 until spamCount and
        hamCount proabilities are called.
        """
        self.name = wordName
        self.count = wordCount
        self.spamCount = 0
        self.hamCount = 0

    def probabilityGivenSpam(self, spamWordCount):
        """
        Calculate the probability this word will appear in a message given that it is spam message by dividing the
        number of times the word appears in spam training messages (spamCount) by the total number of words in the spam
        training data.

        \param spamWordCount (int): The total number of words in the spam training messages
        """
        return float(self.spamCount)/spamWordCount

    def probabilityGivenHam(self, hamWordCount):
        """
        Calculate the probability this word will appear in a message given that it is a ham message by dividing the
        number of times the word appears in ham training messages (hamCount) by the total number of words in the ham
        training data.

        \param hamWordCount (int): the total number of words in the ham training messages
        """
        return float(self.hamCount)/hamWordCount