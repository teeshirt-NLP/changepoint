# Robust Topic-Discriminative word representations

### Introduction

This code implements the algorithm described in [1]. The authors use a statistical model to learn word embeddings that optimally differentiate between topics.

To reproduce the results, see the evaluation script


### How to run the trained model:
1) Use 7-zip (or similar) to decompress the files in [Trained_model/](https://github.com/teeshirt-NLP/changepoint/tree/master/Trained_model)
2) Read in the resulting file containing the 100-dimensional embeddings over 400K words. See the [evaluation script](https://github.com/teeshirt-NLP/changepoint/blob/master/Scripts/4eval.R) for helpful R functions.


### How to train your own embeddings:
1) Download [Wikipedia](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia) then extract it using ```bzip2 -dk enwiki-YOURDATE-pages-articles.xml.bz2```

2) Run the [training scripts](https://github.com/teeshirt-NLP/changepoint/tree/master/Scripts) to create the training and testing datasets. Note: this may take up to 100GB memory and approx 5hrs.

* ```perl 1wikiclean.pl enwiki-latest-pages-articles.xml > enwiki-cleaned-2019```

* ```python 2Makeparagraphdata.py```


3) Install tensorflow==1.14 tensorflow-probability==0.7.0, then run [3train.py](https://github.com/teeshirt-NLP/changepoint/blob/master/Scripts/3train.py)


### How to run the evaluations
🔶  Note: Read through the evaluation script to change path and file names where appropriate.

1) Download the trained model and eval/wikitestdata.csv.zip

2) Step through the [evaluation script](https://github.com/teeshirt-NLP/changepoint/blob/master/Scripts/4eval.R).


### Results
#### Unsupervised evaluation: correlation (lower is better) and modularity (higher is better)

|  | Our method | GloVe |
| --- | --- | --- |
| Median(IQR) correlations | 0.0070 (0.186) |0.109 (0.133) |
| Median(IQR) modularity | 0.306 (0.077) | 0.129 (0.033)  |


#### Supervised evaluation: test set accuracy (higher slope is better)

![text](https://github.com/teeshirt-NLP/changepoint/blob/master/noiseresults.png)


### References
[1] Paper awaiting review

