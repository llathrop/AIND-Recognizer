import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    guesses_other = []
    # TODO implement the recognizer
    # return probabilities, guesses
    
    #for each example in the test set
    for X_test, Xlength_test in list(test_set.get_all_Xlengths().values()):
        word_scores={} #contains the score of each modeled word for this test example
        for word, word_model in models.items(): # use each of the pre-trained word models to score each test example
            try: 
                word_scores[word]=word_model.score(X_test,Xlength_test)
            except: # this word didn't work, give a low score
                word_scores[word]=float("-inf")
                #continue
        probabilities.append(word_scores ) #add to our list of probabilities
        guesses_other=max(word_scores, key=word_scores.get) # from the list of probabilities, store the best guess(highest probability) for each word
        
    #The other thing seems to work, but hold onto this for now
    for current_probability in probabilities:
        guesses.append(max(current_probability, key=current_probability.get))         
        
    if len(guesses) ==len(guesses_other):
        for i in len(guesses_other):
            if not guesses_other[i]==guesses[i]:
                print("broke")
    
    return probabilities,guesses
