# coding: utf-8

import re
import numpy as np
import pandas as pd
import string
import os


          

def mat2channel(x1, x2):
    # takes two arrays of matrices and reshapes to 
    # array of channels of matrices 
    comb=np.array([x1, x2])
    matshape=list(np.shape(x1[1]))
    out=np.reshape(comb, [-1, matshape[0], matshape[1], 2])
    print("output shape: {}".format(np.shape(out)))
    
    return out


   
def generate_complement(X, Y, class_prob):

    # select index set that matches label in Y with p=class_prob
    # and a different label with p=1-p
    new_indicies=[np.random.choice(np.where(Y == x)[0]) 
                 if np.random.uniform() > class_prob 
               else np.random.choice(np.where(Y != x)[0]) for x in Y]

    # generate dataset accordingly
    Ys=np.array([Y[x] for x in new_indicies])
    # one-hot encode classes ([1,0]: match, [0,1]: non-match)
    Ys=np.array([[1,0] if Y[x] == Ys[x] else [0,1] for x in range(len(Y))])
    Xs=np.array([X[x] for x in new_indicies])


    return (Xs, Ys)            


