Training Part of Speech Taggers
-------------------------------

The ``train_tagger.py`` script can use any corpus included with NLTK that implements a ``tagged_sents()`` method. It can also train on the ``timit`` corpus, which includes tagged sentences that are not available through the ``TimitCorpusReader``.

Example usage can be found in `Training Part of Speech Taggers with NLTK Trainer <http://streamhacker.com/2011/03/21/training-part-speech-taggers-nltk-trainer/>`_.

Train the default sequential backoff tagger on the treebank corpus::
	``python train_tagger.py treebank``

To use a brill tagger with the default initial tagger::
	``python train_tagger.py treebank --brill``

To train a NaiveBayes classifier based tagger, without a sequential backoff tagger::
	``python train_tagger.py treebank --sequential '' --classifier NaiveBayes``

To train a unigram tagger::
	``python train_tagger.py treebank --sequential u``

To train on the switchboard corpus::
	``python train_tagger.py switchboard``

To train on a custom corpus, whose fileids end in ".pos", using a `TaggedCorpusReader <http://nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.tagged.TaggedCorpusReader>`_::
	``python train_tagger.py /path/to/corpus --reader nltk.corpus.reader.tagged.TaggedCorpusReader --fileids '.+\.pos'``

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

You can also restrict the files used with the ``--fileids`` option::
	``python train_tagger.py conll2000 --fileids train.txt``

For a complete list of usage options::
	``python train_tagger.py --help``


Using a Trained Tagger
----------------------

You can use a trained tagger by loading the pickle file using `nltk.data.load <http://nltk.org/api/nltk.html#nltk.data.load>`_::
	>>> import nltk.data
	>>> tagger = nltk.data.load("taggers/NAME_OF_TAGGER.pickle")

Or if your tagger pickle file is not in a ``nltk_data`` subdirectory, you can load it with `pickle.load <http://docs.python.org/library/pickle.html#pickle.load>`_::
	>>> import pickle
	>>> tagger = pickle.load(open("/path/to/NAME_OF_TAGGER.pickle"))

Either method will return an object that supports the `TaggerI interface <http://nltk.org/api/nltk.tag.html#nltk.tag.api.TaggerI>`_.

Once you have a ``tagger`` object, you can use it to tag sentences (or lists of words) with the ``tagger.tag(words)`` method::
	>>> tagger.tag(['some', 'words', 'in', 'a', 'sentence'])

``tagger.tag(words)`` will return a list of 2-tuples of the form ``[(word, tag)]``.
