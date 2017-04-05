

# internal modules
from network_elements import *
from two_stream import *
from nlp_preproc import *
# external modules
import pandas as pd
import numpy as np
import gensim


''' 
prepares data for network 
in order to run a training session, use the code in network_elements.py and train.py
tweak the hyperparameters and run '''


# set parameters ------------------------------------
# ---------------------------------------------------

MAXLENGTH=60
EMBEDDING_SIZE=32
MIN_COUNT=1


# read data -----------------------------------------
# ---------------------------------------------------

# i mport csvtable of the form: 
# |other stuff|user1_string|user2_string|other stuff|
indat_pos=pd.read_csv('data/yourdata1.csv')
indat_neg=pd.read_csv('data/yourdata2.csv')
# pos
indat_pos['match']=1
# neg
indat_neg['match']=0


# create dataset ------------------------------------
# ---------------------------------------------------


# reference copy of dataset as a whole
indat=pd.concat([indat_pos[['user1_string', 'user2_string', 'match']]
                 , indat_neg[['user1_string', 'user2_string', 'match']]]
                    ).sample(frac=1).reset_index(drop=True)

# dataset
X1=np.array(cleanup_col(indat.user1_string))
X2=np.array(cleanup_col(indat.user2_string))
Y =np.array(pd.get_dummies(list(indat.match)).as_matrix())


# reduce stringlength -------------------------------
# ---------------------------------------------------


# X1 data
X1=reduce_strings(X1, MAXLENGTH)
# X2 data
X2=reduce_strings(X2, MAXLENGTH)


# preprocess vocab data -----------------------------
# ---------------------------------------------------

# generate word2vec - model
vmodel=genwordvecs(np.concatenate([X1,X2], axis=0), 
                     emb_size=EMBEDDING_SIZE, 
                     try_load=False, 
                     minc=1)

# perform transformation
X1t=w2v_transform(X1, vmodel, MAXLENGTH)  
X2t=w2v_transform(X2, vmodel, MAXLENGTH)  

# final dataset object
data=Dataset(X1t, X2t, Y, testsize=0.2,  shuffle=False)





