Training Classifiers
--------------------

Example usage with the movie_reviews corpus can be found in `Training Binary Text Classifiers with NLTK Trainer <http://streamhacker.com/2010/10/25/training-binary-text-classifiers-nltk-trainer/>`_.

Train a binary NaiveBayes classifier on the movie_reviews corpus, using paragraphs as the training instances:
	``python train_classifier.py --instances paras --classifier NaiveBayes movie_reviews``

Include bigrams as features:
	``python train_classifier.py --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 movie_reviews``

Minimum score threshold:
	``python train_classifier.py --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 --min_score 3 movie_reviews``

Maximum number of features:
	``python train_classifier.py --instances paras --classifier NaiveBayes --ngrams 1 --ngrams 2 --max_feats 1000 movie_reviews``

Use the default Maxent algorithm:
	``python train_classifier.py --instances paras --classifier Maxent movie_reviews``

Use the MEGAM Maxent algorithm:
	``python train_classifier.py --instances paras --classifier MEGAM movie_reviews``

Train on files instead of paragraphs:
	``python train_classifier.py --instances files --classifier MEGAM movie_reviews``

Train on sentences:
	``python train_classifier.py --instances sents --classifier MEGAM movie_reviews``

Evaluate the classifier by training on 3/4 of the paragraphs and testing against the remaing 1/4, without pickling:
	``python train_classifier.py --instances paras --classifier NaiveBayes --fraction 0.75 --no-pickle movie_reviews``

The following classifiers are available:

	* ``NaiveBayes``
	* ``DecisionTree``
	* ``Maxent`` with various algorithms (many of these require `numpy and scipy <http://numpy.scipy.org/>`_, and ``MEGAM`` requires `megam <http://www.cs.utah.edu/~hal/megam/>`_)
	* ``Svm`` (requires `svmlight <http://svmlight.joachims.org/>`_ and `pysvmlight <https://bitbucket.org/wcauchois/pysvmlight>`_)

If you also have `scikit-learn <http://scikit-learn.org/>`_ then the following classifiers will also be available, with ``sklearn`` specific training options. If there is a sklearn classifier or training option you want that is not present, please `submit an issue <https://github.com/japerk/nltk-trainer/issues>`_.

	* `sklearn.ExtraTreesClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier>`_
	* `sklearn.GradientBoostingClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier>`_
	* `sklearn.RandomForestClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier>`_
	* `sklearn.LogisticRegression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`_
	* `sklearn.BernoulliNB <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB>`_
	* `sklearn.GaussianNB <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB>`_
	* `sklearn.MultinomialNB <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB>`_
	* `sklearn.KNeighborsClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier>`_
	* `sklearn.LinearSVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_
	* `sklearn.NuSVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVC.html#sklearn.svm.NuSVC>`_
	* `sklearn.SVC <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC>`_
	* `sklearn.DecisionTreeClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier>`_

For example, here's how to use the ``sklearn.LinearSVC`` classifier with the ``movie_reviews`` corpus:
	``python train_classifier.py movie_reviews --classifier sklearn.LinearSVC``

For a complete list of usage options:
	``python train_classifier.py --help``


Using a Trained Classifier
--------------------------

You can use a trained classifier by loading the pickle file using `nltk.data.load <http://nltk.org/api/nltk.html#nltk.data.load>`_:
	>>> import nltk.data
	>>> classifier = nltk.data.load("classifiers/NAME_OF_CLASSIFIER.pickle")

Or if your classifier pickle file is not in a ``nltk_data`` subdirectory, you can load it with `pickle.load <http://docs.python.org/library/pickle.html#pickle.load>`_:
	>>> import pickle
	>>> classifier = pickle.load(open("/path/to/NAME_OF_CLASSIFIER.pickle"))

Either method will return an object that supports the `ClassifierI interface <http://nltk.org/api/nltk.classify.html#nltk.classify.api.ClassifierI>`_. 

Once you have a ``classifier`` object, you can use it to classify word features with the ``classifier.classify(feats)`` method, which returns a label:
	>>> words = ['some', 'words', 'in', 'a', 'sentence']
	>>> feats = dict([(word, True) for word in words])
	>>> classifier.classify(feats)

If you used the ``--ngrams`` option with values greater than 1, you should include these ngrams in the dictionary using `nltk.util.ngrams(words, n) <http://nltk.org/api/nltk.html#nltk.util.ngrams>`_:
	>>> from nltk.util import ngrams
	>>> words = ['some', 'words', 'in', 'a', 'sentence']
	>>> feats = dict([(word, True) for word in words + ngrams(words, n)])
	>>> classifier.classify(feats)

The list of words you use for creating the feature dictionary should be created by `tokenizing <http://text-processing.com/demo/tokenize/>`_ the appropriate text instances: sentences, paragraphs, or files depending on the ``--instances`` option.

