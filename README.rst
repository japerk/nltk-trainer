NLTK Trainer
------------

NLTK Trainer exists to make training and evaluating NLTK objects as easy as possible.


Requirements
------------

You must have Python 2.6 with `argparse <http://docs.python.org/library/argparse.html>`_ and `NLTK <http://www.nltk.org/>`_ 2.0 installed. `NumPy <http://numpy.scipy.org/>`_, `SciPy <http://www.scipy.org/>`_, and `megam <http://www.cs.utah.edu/~hal/megam/>`_ are recommended for training Maxent classifiers.


Training Classifiers
--------------------

For a complete list of usage options::
	python train_classifier.py --help

Train a binary NaiveBayes classifier on the movie_reviews corpus, using paragraphs as the training instances::
	python train_classifier.py --instances paras --algorithm NaiveBayes movie_reviews

Include bigrams as features::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams movie_reviews

Minimum score threshold::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams --min_score 3 movie_reviews

Maximum number of features::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams --max_feats 1000 movie_reviews

Use the default Maxent algorithm::
	python train_classifier.py --instances paras --algorithm Maxent movie_reviews

Use the MEGAM Maxent algorithm::
	python train_classifier.py --instances paras --algorithm MEGAM movie_reviews

Train on files instead of paragraphs::
	python train_classifier.py --instances files --algorithm MEGAM movie_reviews

Train on sentences::
	python train_classifier.py --instances sents --algorithm MEGAM movie_reviews

Evaluate the classifier by training on 3/4 of the paragraphs and testing against the remaing 1/4, without pickling::
	python train_classifier.py --instances paras --algorithm NaiveBayes --fraction 0.75 --no-pickle movie_reviews