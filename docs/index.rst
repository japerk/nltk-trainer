Welcome to NLTK-Trainer's documentation!
========================================

*NLTK-Trainer* is a set of `Python <http://www.python.org/>`_ command line scripts for natural language processing. With these scripts, you can do the following things without writing a single line of code:

1. train `NLTK <http://nltk.org/>`_ based models
2. evaluate pickled models against a corpus
3. analyze a corpus

These scripts are Python 2 & 3 compatible and work with NLTK 2.0.4 and higher.

Download
========

The scripts can be downloaded from `nltk-trainer <https://github.com/japerk/nltk-trainer>`_ on github.

Documentation
=============

.. toctree::
   :maxdepth: 2
   
   train_classifier.rst
   train_tagger.rst
   train_chunker.rst
   analyze_tagged_corpus.rst
   analyze_tagger_coverage.rst

Books
=====

.. image:: http://ws-na.amazon-adsystem.com/widgets/q?_encoding=UTF8&ASIN=1782167854&Format=_SL160_&ID=AsinImage&MarketPlace=US&ServiceVersion=20070822&WS=1&tag=streamhacker-20
	:target: http://www.amazon.com/gp/product/1782167854/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1782167854&linkCode=as2&tag=streamhacker-20&linkId=K2BYHHUBZ4GIEW4L
	:alt: Python 3 Text Processing with NLTK 3 Cookbook

`Python 3 Text Processing with NLTK 3 Cookbook <http://www.amazon.com/gp/product/1782167854/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1782167854&linkCode=as2&tag=streamhacker-20&linkId=K2BYHHUBZ4GIEW4L>`_ contains many examples for training NLTK models with & without NLTK-Trainer. 

* Chapter 4 covers part-of-speech tagging and :ref:`train_tagger.py <train_tagger>`.
* Chapter 5 shows how to train phrase chunkers and use :ref:`train_chunker.py <train_chunker>`.
* Chapter 7 demonstrates classifier training and :ref:`train_classifier.py <train_classifier>`.

Articles
========

- `Training Binary Classifiers with NLTK Trainer <http://streamhacker.com/2010/10/25/training-binary-text-classifiers-nltk-trainer/>`_
- `Training Part of Speech Taggers with NLTK Trainer <http://streamhacker.com/2011/03/21/training-part-speech-taggers-nltk-trainer/>`_
- `Analyzing Tagger Corpora and NLTK Part of Speech Taggers <http://streamhacker.com/2011/03/23/analyzing-tagged-corpora-nltk-part-speech-taggers/>`_
- `NLTK Default Tagger Coverage of treebank corpus <http://streamhacker.com/2011/01/24/nltk-default-tagger-treebank-tag-coverage/>`_
- `NLTK Default Tagger Coverage of conll2000 corpus <http://streamhacker.com/2011/01/25/nltk-default-tagger-conll2000-tag-coverage/>`_

Demos and APIs
==============

Nearly all the models that power the `text-processing.com <http://text-processing.com/>`_ NLTK demos and NLP APIs have been trained using NLTK-Trainer.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

