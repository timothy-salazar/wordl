""" This is to dunk on my mom
"""
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

def get_5():
    """ Input:
            None
        Output:
            returns a numpy array of 5 letter words that contain only
            alphabetic characters
    """
    return np.array([i.lower() for i in wn.words() if len(i) == 5 and i.isalpha()])

class WordlGuess():
    """ Keeps track of our gueses, updates stats
    """
    def __init__(self):
        self.corpus = get_5()
        self.update_freq()
        self.eliminated_letters = set()
        self.wrong_location = dict()
        self.right_location = dict()
        self.verbose = True

    def update_freq(self):
        """ counts the letters in the remaining words in our corpus.
        Saves a dictionary mapping letter to frequency (count of letter / total letters)
        """
        letters = [letter for word in self.corpus for letter in word]
        freq_dist = nltk.FreqDist(letters)
        self.freq = {k:freq_dist.freq(k) for k in freq_dist.keys()}

    def update_corpus(self):
        """ This updates the corpus by checking each word to see if it's been eliminated
        by some clue we received since the last time we checked
        """
        corpus_length = len(self.corpus)
        self.corpus = np.array([word for word in self.corpus if self.check_word_legality(word)])
        if self.verbose:
            print(f'Corpus length went from {corpus_length} to {len(self.corpus)}')
        self.update_freq()
        self.sort_corpus()

    def check_word_legality(self, word):
        """ Input:
                word: string - the word we're checking
            Output:
                bool - whether the word has been eliminated from consideration or not
        This checks the word against the clues we've amassed. The word will be eliminated if:
            - it contains any letters in self.eliminated_letters
            - it contains a letter in a position which we already know that letter does not
            appear in
            - it does not contain each letter whose position we know with certainty in those
            positions
        """
        # check whether the word contains any of the letters we've determined will
        # not appear in the word
        if set(word).intersection(self.eliminated_letters):
            return False
        # check to see whether the word contains any letters in a position where we already
        # know that letter does not appear
        for letter, position_list in self.wrong_location.items():
            for position in position_list:
                if word[position] == letter:
                    return False
        # make sure that each letter we've know the location of appears in the right position
        for letter, position in self.right_location.items():
            if word[position] != letter:
                return False
        return True

    def sort_corpus(self):
        """ calulates a score for each word. This score is calculated by taking each letter in
        the word, taking its frequency (larger frequencies indicate that the letter appears more
        often in the corpus), and then summing these frequencies.
        We then sort the corpus from higher score to lower score.
        """
        word_scores = [sum([self.freq[letter] for letter in word]) for word in self.corpus]
        # we have to add the [::-1] because numpy's argsort returns an array that will sort
        # from small to large
        self.corpus = self.corpus[np.argsort(word_scores)[::-1]]

    def yellow(self, letter, position):
        """ Input:
                letter: string - a letter
                position: int - the index of the letter in our guess
        "letter" and "position" give us a guess. This function assumes that this guess was yellow -
        that is, it's a letter in our word, but the position is incorrect.
        
        This will update self.wrong_location - a dictionary with letters for keys and a list of
        positions for values. If there is no entry in self.wrong_location for "letter", we make a
        list and add "position" - otherwise we add "position" to the end of the list
        """
        self.wrong_location[letter] = self.wrong_location.get(letter, []) + [position]
    
    def green(self, letter, position):
        """ Input:
                letter: string - a letter
                position: int - the index of the letter in our guess
        "letter" and "position" give us a guess. This function assumes that this guess was green -
        that is, the letter and position were correct

        This will update self.right location, which maps the letters we KNOW are in the word
        to their position within the string
        """
        if letter in self.wrong_location:
            self.wrong_location.pop(letter)
        self.right_location[letter] = position

    def grey(self, letters):
        """ Input:
                letter: string - a letter or string
        This function assumes that we guessed a letter and were told that it does not
        appear in the word in any location
        """
        for letter in letters:
            self.eliminated_letters.add(letter)
