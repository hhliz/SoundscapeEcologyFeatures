#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liz M. Huancapaza Hilasaca
# Copyright (c) 2021
# E-mail: lizhh@usp.br

# pandas 
# sklearn
# matplotlib

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

# import multiprocessing as mp
from multiprocessing import Pool, Manager, Process, Lock
# from multiprocessing import 


from sklearn.manifold import TSNE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

class ModelFeature:

    def __init__(self, datsetfile, target):
        df = pd.read_csv(datsetfile, delimiter=",")
        self.featurenames = []
        self.target = target
        for c in df.columns:
            if c != target:
                self.featurenames.append(c)
        self.featurenamesi = {self.featurenames[i]:i for i in range(len(self.featurenames))}
        
        manager = Manager()
        self.acc = manager.list([0.0 for i in range(len(self.featurenames))])
        
        # self.X = df[self.featurenames].values.tolist()
        # self.y = df[self.target].values.tolist()

        self.X = df[self.featurenames].to_numpy()
        self.y = df[self.target].to_numpy()
        # X = np.array(self.X)
        # y = np.array(self.y)

        # self.idxbest_w = manager.Value("i",-1.0)
        # self.idxbest_i = manager.Value("i",-1)

        self.besfe={}

        # print (self.X)


    def normalization(self, nodel):
        if nodel=="zscore":
            # self.X = df.readcsv(datsetfile)
            self.X = (self.X - self.X.mean())/self.X.std()

    def rankingFeatures(self):
        X = (self.X)
        y = (self.y)

        model = ExtraTreesClassifier(n_estimators=100, random_state=7)
        model.fit(X, y)

        imp = sorted(zip(self.featurenames, model.feature_importances_), key=lambda x: x[1] * -1)
        # kys = []
        # vls = []
        rk = []
        for r in imp:
            # kys.append(r[0])
            # vls.append(r[1])
            rk.append({"key":r[0], "value":r[1]})

        return rk

    def selectColumn(self, rfe):
        XC = np.zeros((len(self.X), len(rfe)) )
        for ii in range(len(self.X)):
            jx = 0
            for e in rfe:
                jj = self.featurenamesi[e["key"]]
                XC[ii, jx] = self.X[ii,jj]
                jx = jx+1
        return XC

    def bestfeatures(self, rfe):
        idex = np.array([i for i in range(len(self.y))])
        Xi = idex

        k = 5
        kf = KFold(n_splits=k, random_state=7, shuffle=True)
        spldata = []

        for train_ix, test_ix in kf.split(Xi):
            spldata.append([Xi[train_ix], Xi[test_ix]])

        accs = [ 0.0 for i in range(len(rfe))]
        idxbest = -1
        idxbest_w = 0.0




        aux = []
        for i in range(len(rfe)):
            aux.append({"rfe":rfe[0:i+1], "spldata":spldata, "i":i})
            # XC = self.selectColumn(rfe[0:i+1])
            # # XC = np.array(XC) 
            # # print(XC)
            # outcomes = []
            # outcomeslabels_train = []
            # outcomeslabels_test = []
            # model = RandomForestClassifier(n_estimators=100, random_state=7)
            # for train_index, test_index in spldata:
            #     Xtrain, Xtest = XC[train_index], XC[test_index]
            #     ytrain, ytest = self.y[train_index], self.y[test_index]
            #     model.fit(Xtrain, ytrain)
            #     predictions = model.predict(Xtest)

            #     accuracy = accuracy_score(ytest, predictions)
            #     outcomes.append(accuracy)
            #     outcomeslabels_train.append([test_index, ytest, predictions])
                
            # outcomes = np.array(outcomes)
            # accvalidate_mean = outcomes.mean()
            # accvalidate_std= outcomes.std()

            # if accs[i]>idxbest_w:
            #     idxbest_w = accs[i]
            #     idxbest=i
            # print("accvalidate_mean", accvalidate_mean)

        pool = Pool(processes=7)
        rs = pool.map(self.processes, aux)
        pool.close()

        idxbest_w = -1.0
        idxbest_i = -1
        for i in range(len(self.acc)):
            if self.acc[i]>idxbest_w:
                idxbest_w = self.acc[i]
                idxbest_i = i

        self.besfe = rfe[:idxbest_i+1]

        # self.besfe = rfe[:self.idxbest_i+1]
        # print("XXXXX", self.acc)
        # print("self.besfe", self.besfe)
        # print("iiiiiiiii", idxbest_i)
        return self.besfe

    def tsne(self, X):
        X2 = TSNE(n_components=2, random_state=0, perplexity=40).fit_transform(X)
        return X2.tolist()
        
    def plot2D(self, rfe, model, namefile):
        fig, ax = plt.subplots(figsize=(5, 5))
        if model == "TSNE":
            X = self.selectColumn(rfe)
            X2 = self.tsne(X)
            cmap = plt.cm.get_cmap('rainbow')
            # norm = matplotlib.colors.Normalize(vmin=0,vmax=10)
            # colors = [matplotlib.colors.rgb2hex( cmap(norm(ij))[:3] ) for ij in range(10)]
            colors = ["#ff0000","#00ff00","#0000ff","#ffff00"]
            for i in range(len(self.y)):
                ax.scatter( X2[i][0], X2[i][1],
                            c=colors[int(self.y[i])], s=7,
                            alpha=0.85, linewidth=0.0001,
                            marker="o")

        fig = ax.get_figure()
        fig.savefig(namefile, format="pdf", bbox_inches='tight')
        plt.close("all")

    def processes(self, row):

        rfe = row["rfe"]
        spldata = row["spldata"]
        i = row["i"]

        XC = self.selectColumn(rfe)
        outcomes = []
        outcomeslabels_train = []
        outcomeslabels_test = []
        model = RandomForestClassifier(n_estimators=100, random_state=7)
        for train_index, test_index in spldata:
            Xtrain, Xtest = XC[train_index], XC[test_index]
            ytrain, ytest = self.y[train_index], self.y[test_index]
            model.fit(Xtrain, ytrain)
            predictions = model.predict(Xtest)

            accuracy = accuracy_score(ytest, predictions)
            outcomes.append(accuracy)
            outcomeslabels_train.append([test_index, ytest, predictions])
                
        outcomes = np.array(outcomes)
        self.acc[i] = outcomes.mean()
        accvalidate_std= outcomes.std()

        # if self.acc[i]>self.idxbest_w:
        #     self.idxbest_w = self.acc[i]
        #     self.idxbest_i = i
        print("acc_mean", i, self.acc[i])

