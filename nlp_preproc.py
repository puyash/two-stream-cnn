
import re
import numpy as np
import pandas as pd
import string
#import os
import gensim

def cleanup_str(st, numbers=False):
    
    st=str(st)

    if numbers == True:
        keep=set(string.letters + string.digits + ' ')    
    else: 
        keep=set(string.letters + ' ')
    
    # clean string
    st=''.join(x if x in keep else ' ' for x in st)
    # rem multiple spaces
    st=re.sub(' +',' ', st)

    return st.strip().lower()   

# mapper: cleanup a pd column or list of strings
def cleanup_col(col, numbers=False):
    
    col=map(lambda x: cleanup_str(x, numbers=numbers), col)
    return col
    

# perform cleanup on a pandas dataframe or dict
def cleanup_frame(dat, collist=['user1_string', 'user2_string'], numbers=False):
    for col in collist:
        dat[col]=cleanup_col(dat[col], numbers=False)
        
    return dat

# shorten strings in list
def reduce_strings(stringlist, maxlength, return_arrays=True):
    
    # if type(stringlist) != list:
    #    stringlist=list(stringlist)
    
    splitsreduce=[x[0:maxlength] for x in [x.split(' ') for x in stringlist]]
    
    if return_arrays:
        return splitsreduce
    
    shortstrings=[' '.join(x) for x in splitsreduce]
    return shortstrings



def vec2pad(doc, max_length):
    
    doclength, embdim=np.shape(doc)
    # add zeros up the decided sequence length 
    if doclength < max_length:
        s=np.zeros([max_length - doclength, embdim]) 
        doc=np.concatenate((doc,s), axis=0)
        
        return doc
    elif doclength == max_length:
        
        return doc
    else: 
        print("document is longer that the set max_length")
        
        return doc


            
def str_length_distr(string_arrays, max_length=40):
    # strings of usererdata and their respective lengths

    stringlengths=[len(x.split(' ')) for x in string_arrays]
    
    # maximum length of user string
    maxstringlength=max(stringlengths)
    print("maximum string length: {}".format(maxstringlength))
    numchanged=len([x for x in stringlengths if x >= max_length])
    print("number docs changed with cap at {}: {} of total {} ({}%)".format(
        max_length, numchanged, len(string_arrays), np.round(100.0*numchanged/len(string_arrays), 2)))
    # plot length distribution
    plt.hist(stringlengths)  


def genwordvecs(docs, emb_size, try_load=False, minc=1):

    vmodel_name='embedding_dim_{}_c_{}'.format(emb_size, minc)
    if try_load == True:
        try: 
            vmodel=gensim.models.Word2Vec.load('vmodels/'+vmodel_name)
            print('model loaded from disk')
            
            return vmodel
        except IOError:
            print('error loading model..')
            print('training word embeddings')

    vmodel=gensim.models.Word2Vec(docs, min_count=minc, size=emb_size, workers=4)
    vmodel.save('vmodels/'+vmodel_name)
    
    return vmodel        




def w2v_transform(string_arrays, model, max_length=None):
    
    # removes words that not in vocabulary and then transforms to vector form
    v2w_arrays=map(lambda x: model[[y for y in x if y in model]], string_arrays)
    # sets length limit and zero-vectors as padding
    if max_length != None:
        v2w_arrays=map(lambda x: vec2pad(x, max_length), v2w_arrays)
        
    return np.array(v2w_arrays)
   


