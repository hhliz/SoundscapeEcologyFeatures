
# Visualization and categorization of ecological acoustic events based on discriminant features
 [doi: https://doi.org/10.1016/j.ecolind.2020.107316](https://doi.org/10.1016/j.ecolind.2020.107316)

Although sound classification in soundscape studies are generally performed by experts, the large growth of acoustic data presents a major challenge for performing such task. At the same time, the identification of more discriminating features becomes crucial when analyzing soundscapes, and this occurs because natural and anthropogenic sounds are very complex, particularly in Neotropical regions, where the biodiversity level is very high. In this scenario, the need for research addressing the discriminatory capability of acoustic features is of utmost importance to work towards automating these processes. In this study we present a method to identify the most discriminant features for categorizing sound events in soundscapes. Such identification is key to classification of sound events. Our experimental findings validate our method, showing high discriminatory capability of certain extracted features from sound data, reaching an accuracy of 89.91% for classification of frogs, birds and insects simultaneously. An extension of these experiments to simulate binary classification reached accuracy of 82.64%, 100%, 99.40% and  for the classification between combinations of frogs-birds, frogs-insects and birds-insects, respectively.

## Authors:

   Liz Huancapaza (University of Sao Paulo, Brazil)\
   Lucas Gaspar (Sao Paulo State University - UNESP, Brazil)\
   Milton Ribeiro (Sao Paulo State University - UNESP, Brazil)\
   Rosane Minghim (University College Cork, Ireland)

## Running   
* python3 Main.py

## How use
```python
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
    model = ModelFeature("datas/sound_o_anuros_aves.csv","TARGET_")

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
```

### Audio set used in this paper
[For this paper](https://doi.org/10.1016/j.ecolind.2020.107316) our audio set comprised 2,277 sound files of one-minute each, divided into three classes.

* [Soundscape audio set](https://github.com/LEEClab/soundscape_CCM1_exp01)
  * Content: Soundscape ecology. Divided into three classes: 615 for frogs, 822 for birds and 840 for insects.
  * Sound files of one-minute each.


### Sample data set with features description processed

The descriptors used for feature extraction were presented [in this paper](https://doi.org/10.1016/j.ecolind.2020.107316) and were categorized into three groups. As a result, 238 features were processed for each audio minute file.

* [Soundscape dataset - Features Extracted](https://github.com/hhliz/SoundscapeEcologyFeatures/tree/master/datas)
  * 238 features (categorized into three groups: based on images; based on spectrum; and based on acoustic indices)
  * Features extracted of one-minute each.




## Citation
<pre><code>
@article{huancapaza2021ecofeatures,
author = "Liz Maribel {Huancapaza Hilasaca} and Lucas Pacciullio Gaspar and Milton Cezar Ribeiro 
and Rosane Minghim",
title = "Visualization and categorization of ecological acoustic events based on discriminant features",
journal = "Ecological Indicators",
pages = "107316",
year = "2021",
issn = "1470-160X",
url = "http://www.sciencedirect.com/science/article/pii/S1470160X20312589",

abstract = "Although sound classification in soundscape studies are generally performed by experts, 
the large growth of acoustic data presents a major challenge for performing such task. At the same 
time, the identification of more discriminating features becomes crucial when analyzing soundscapes, 
and this occurs because natural and anthropogenic sounds are very complex, particularly in Neotropical 
regions, where the biodiversity level is very high. In this scenario, the need for research addressing 
the discriminatory capability of acoustic features is of utmost importance to work towards automating 
these processes. In this study we present a method to identify the most discriminant features for 
categorizing sound events in soundscapes. Such identification is key to classification of sound events. 
Our experimental findings validate our method, showing high discriminatory capability of certain extracted 
features from sound data, reaching an accuracy of 89.91% for classification of frogs, birds and insects 
simultaneously. An extension of these experiments to simulate binary classification reached accuracy of 
82.64%,100.0% and 99.40% for the classification between combinations of frogs-birds, frogs-insects and 
birds-insects, respectively.",
keywords = "Soundscape ecology, Discriminant features, Visualization, Classification, Feature selection",
doi = "https://doi.org/10.1016/j.ecolind.2020.107316"
}
}</code></pre>


## Get in touch
Liz Huancapaza - lizhh@usp.br
