Training Classifiers
--------------------

Example usage with the movie_reviews corpus can be found in `Training Binary Text Classifiers with NLTK Trainer <http://streamhacker.com/2010/10/25/training-binary-text-classifiers-nltk-trainer/>`_.

Train a binary NaiveBayes classifier on the movie_reviews corpus, using paragraphs as the training instances::
	``python train_classifier.py --instances paras --classifier NaiveBayes movie_reviews``

Include bigrams as features::
	``python train_classifier.py --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 movie_reviews``

Minimum score threshold::
	``python train_classifier.py --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 --min_score 3 movie_reviews``

Maximum number of features::
	``python train_classifier.py --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 --max_feats 1000 movie_reviews``

Use the default Maxent algorithm::
	``python train_classifier.py --instances paras --classifier Maxent movie_reviews``

Use the MEGAM Maxent algorithm::
	``python train_classifier.py --instances paras --classifier MEGAM movie_reviews``

Train on files instead of paragraphs::
	``python train_classifier.py --instances files --classifier MEGAM movie_reviews``

Train on sentences::
	``python train_classifier.py --instances sents --classifier MEGAM movie_reviews``

Evaluate the classifier by training on 3/4 of the paragraphs and testing against the remaing 1/4, without pickling::
	``python train_classifier.py --instances paras --classifier NaiveBayes --fraction 0.75 --no-pickle movie_reviews``

For a complete list of usage options::
	``python train_classifier.py --help``


Using a Trained Classifier
--------------------------

You can use a trained classifier by loading the pickle file using `nltk.data.load <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.data-module.html#load>`_::
	>>> import nltk.data
	>>> classifier = nltk.data.load("classifiers/NAME_OF_CLASSIFIER.pickle")

Or if your classifier pickle file is not in a ``nltk_data`` subdirectory, you can load it with `pickle.load <http://docs.python.org/library/pickle.html#pickle.load>`_::
	>>> import pickle
	>>> classifier = pickle.load(open("/path/to/NAME_OF_CLASSIFIER.pickle"))

Either method will return an object that supports the `ClassifierI interface <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.classify.api.ClassifierI-class.html>`_. 

Once you have a ``classifier`` object, you can use it to classify word features with the ``classifier.classify(feats)`` method, which returns a label::
	>>> words = ['some', 'words', 'in', 'a', 'sentence']
	>>> feats = dict([(word, True) for word in words])
	>>> classifier.classify(feats)

If you used the ``--ngrams`` option with values greater than 1, you should include these ngrams in the dictionary using `nltk.util.ngrams(words, n) <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.util-module.html#ngrams>`_::
	>>> from nltk.util import ngrams
	>>> words = ['some', 'words', 'in', 'a', 'sentence']
	>>> feats = dict([(word, True) for word in words + ngrams(words, n)])
	>>> classifier.classify(feats)

The list of words you use for creating the feature dictionary should be created by `tokenizing <http://text-processing.com/demo/tokenize/>`_ the appropriate text instances: sentences, paragraphs, or files depending on the ``--instances`` option.

