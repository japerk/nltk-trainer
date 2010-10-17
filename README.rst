--------------------
Training Classifiers
--------------------

To train a binary NaiveBayes classifier on the movie_reviews corpus, using paragraphs as the training instances ::
	python train_classifier.py --instances paras --algorithm NaiveBayes movie_reviews

To also include bigrams as features:
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams movie_reviews

To set a minimum score threshold::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams --min_score 3 movie_reviews

To set a maximum number of features::
	python train_classifier.py --instances paras --algorithm NaiveBayes --bigrams --max_feats 1000 movie_reviews

To use the default Maxent algorithm::
	python train_classifier.py --instances paras --algorithm Maxent movie_reviews

To use the default MEGAM Maxent algorithm::
	python train_classifier.py --instances paras --algorithm MEGAM movie_reviews

To train on files instead of paragraphs::
	python train_classifier.py --instances files --algorithm MEGAM movie_reviews

To train on sentences::
	python train_classifier.py --instances sents --algorithm MEGAM movie_reviews