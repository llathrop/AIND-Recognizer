import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("+inf")
        best_model = None
        num_features = self.X.shape[1]
        
        for curr_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model=self.base_model(curr_components) 
                logL=model.score(self.X, self.lengths)
                logN=np.log(len(self.X))
                
                # p = Initial state occupation probabilities + Transition probabilities + Emission probabilities
                # this formula for p comes via the udacity forums:
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/4
                #Initial_state_occupation_probabilities = curr_components
                #Transition_probabilities = curr_components*(curr_components - 1)
                #Emission_probabilities = curr_components*num_features*2 
                #p = Initial_state_occupation_probabilities + Transition_probabilities + Emission_probabilities
                
                free_transition_probability = curr_components*(curr_components-1)
                free_starting_probabilities = curr_components-1 
                Number_of_means = curr_components*num_features
                Number_of_covariances=curr_components*num_features
                
                p = free_transition_probability + free_starting_probabilities + Number_of_means + Number_of_covariances
                
                BIC = -2 * logL + p * logN
                
                if BIC<best_score: #A higher BIC score indicates a better fit
                    best_score=BIC
                    best_model=model
            except: # if we fail at any point we don't want to modify the best score/model
                continue
        return best_model
            
        
        best_num_components = self.n_constant
        return self.base_model(best_num_components)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = log_other_words_score=float("-inf")
        best_model = None

        for curr_components in range(self.min_n_components, self.max_n_components+1):
            try:
                model=self.base_model(curr_components)
                logL=model.score(self.X, self.lengths)
                sum_other_words_score=0
                count_other_words=0
                
                for curr_word in self.words:
                    if curr_word not in self.this_word:
                        curr_word_X, curr_word_length=self.hwords[curr_word]
                        try:
                            curr_word_score=model.score(curr_word_X, curr_word_length)       
                            sum_other_words_score= sum_other_words_score+curr_word_score
                            count_other_words+=1
                        except:                                                        
                            continue
                if count_other_words>0:
                    log_other_words_score=sum_other_words_score/count_other_words
                
                DIC=logL-log_other_words_score
                
                if DIC>best_score:
                    best_score=DIC
                    best_model=model    
            except: 
                continue
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_model=None
        best_model_score=model_score=float("-inf")
        num_splits= min(3, len(self.lengths))
        
        if num_splits ==1:
            #print("One component, return base_model")
            return self.base_model(num_splits)
            
        for curr_components in range(self.min_n_components, self.max_n_components+1):
            split_method = KFold(n_splits=num_splits)
            num_splits_tried=0
            split_score=0
            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                X_train, y_train = combine_sequences(cv_train_idx, self.sequences)
                curr_model=None                
                try:
                    curr_model=GaussianHMM(n_components=curr_components, covariance_type="diag", n_iter=1000, 
                                           random_state=self.random_state, verbose=False).fit(X_train, y_train)
                    num_splits_tried+=1
                    split_score= split_score+curr_model.score(X_train, y_train)
                    #print("tried and it worked!",curr_components,split_score,num_splits_tried)
                except:
                    continue
            if num_splits_tried>0:
                model_score=split_score/num_splits
            
            if model_score> best_model_score:
                #print("found a better model")
                best_model_score=model_score
                best_model=curr_model
        #if best_model==None:
            #print("none shall pass")
            #return self.base_model(num_splits)
        return best_model
                   
    