# Topic-Discriminative word representations

### Introduction

This code implements the algorithm described in [1]. The authors use a statistical model to learn word embeddings that optimally differentiate between topics.

To reproduce the results, see the evaluation script


### How to run the trained model:
1) Use 7-zip (or similar) to decompress the files in [Trained_model/](https://github.com/osotsia/BOCE-embeddings/tree/master/Trained_model)
2) Read in the resulting file containing the 100-dimensional embeddings over 400K words. See the evaluation scripts for helpful R functions.


### How to train your own embeddings:
1) [Download](https://en.wikipedia.org/wiki/Wikipedia:Database_download) Wikipedia then extract it using ```bzip2 -dk enwiki-YOURDATE-pages-articles.xml.bz2```

2) Run the Training scripts to create the training and testing datasets. Note: this may take up to 100GB memory and approx 5hrs.

* ```perl 1wikiclean.pl enwiki-latest-pages-articles.xml > enwiki-cleaned-2019```

* ```python 2Makeparagraphdata.py```


3) Install tensorflow==1.14 tensorflow-probability==0.7.0, then run 3train.py

🔶 Note: Read through the scripts to change path and file names particularly for the evaluation, as it requires downloading GloVe vectors.

### Results



### References


