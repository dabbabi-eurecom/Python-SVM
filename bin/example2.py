#!/usr/bin/env python

import svmpy
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import argh
import sys


def example(num_samples=8, num_features=2, grid_size=20, filename="example2.pdf"):
    l=[2.0,1.0,2.0,2.0,3.0,2.0,1.0,3.0,0.0,1.0,-1.0,-1.0,-1.0,-2.0,-2.0,-1.0]  # here put your input training vectors
    lmat=np.asarray(l)
    lab=[1.0,1.0,1.0,1.0,-1.0,-1.0,-1.0,-1.0] # here put your labels 
    llab=np.asarray(lab)
    
    samples = np.matrix(lmat # defining training matrix
                        .reshape(num_samples, num_features)) # re-shaping the matrix 
    
    labels = np.matrix(llab # defining training matrix
                        .reshape(num_samples, 1)) # re-shaping the matrix
   
    trainer = svmpy.SVMTrainer(svmpy.Kernel.linear(), 0.1) # linear kernel 
    predictor = trainer.train(samples, labels)

    plot(predictor, samples, labels, grid_size, filename)

    # predict 
    print "training performed , predict !"
    i=raw_input("space seperated coordinates of prediction vector : ")
    try:
        s=i.split()
        x=[float(s[0]),float(s[1])]
    except:
        sys.exit(0)
    try:
        xmat=np.asarray(x)
        xp=np.matrix(xmat # defining training matrix
                        .reshape(1, num_features)) # re-shaping the predict input 
        print predictor.predict(xp)
    except:
         sys.exit(0)

def plot(predictor, X, y, grid_size, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1,)

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        result.append(predictor.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.savefig(filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    argh.dispatch_command(example)
