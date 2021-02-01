#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Liz M. Huancapaza Hilasaca
# Copyright (c) 2021
# E-mail: lizhh@usp.br

from ModelFeature import *

if __name__ == "__main__":
    """
    The methodology
    """
    #(1) Data Pre-processing
    #(2) Feature description
    #(3) Discriminant feature analysis
    #(3.1) Data cleaning
    #(3.2) Normalization
    #(3.3) Feature analysis
    #(3.4) Feature selecction
    #(3.5) Identification of the most discriminating features
    #(4) Visualization



    #######################
    # begin
    #######################

    #(1) Data Pre-processing
    #(2) Feature description
    #(3) Discriminant feature analysis

    #(3.1) Data cleaning
    model = ModelFeature("datasets/sound_o_anuros_aves.csv","TARGET_")

    #(3.2) Normalization
    model.normalization("zscore")

    #(3.3) Feature analysis
    #######################

    #(3.4) Feature selecction   
    RF = model.rankingFeatures()

    #(3.5) Identification of the most discriminating features
    BF = model.bestfeatures(RF)

    #(4) Visualization
    model.plot2D(BF, "TSNE", "plot.pdf")

    #######################
    # end
    #######################






