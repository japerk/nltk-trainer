Training IOB Chunkers
---------------------

The ``train_chunker.py`` script can use any corpus included with NLTK that implements a ``chunked_sents()`` method.

Train the default sequential backoff tagger based chunker on the treebank_chunk corpus::
	``python train_chunker.py treebank_chunk``

To train a NaiveBayes classifier based chunker::
	``python train_chunker.py treebank_chunk --classifier NaiveBayes``

To train on the conll2000 corpus::
	``python train_chunker.py conll2000``

To train on a custom corpus, whose fileids end in ".pos", using a `ChunkedCorpusReader <http://nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.ChunkedCorpusReader>`_::
	``python train_chunker.py /path/to/corpus --reader nltk.corpus.reader.chunked.ChunkedCorpusReader --fileids '.+\.pos'``

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

You can also restrict the files used with the ``--fileids`` option::
	``python train_chunker.py conll2000 --fileids train.txt``

For a complete list of usage options::
	``python train_chunker.py --help``


Using a Trained Chunker
-----------------------

You can use a trained chunker by loading the pickle file using `nltk.data.load <http://nltk.org/api/nltk.html#nltk.data.load>`_::
	>>> import nltk.data
	>>> tagger = nltk.data.load("chunkers/NAME_OF_CHUNKER.pickle")

Or if your chunker pickle file is not in a ``nltk_data`` subdirectory, you can load it with `pickle.load <http://docs.python.org/library/pickle.html#pickle.load>`_::
	>>> import pickle
	>>> tagger = pickle.load(open("/path/to/NAME_OF_CHUNKER.pickle"))

Either method will return an object that supports the `ChunkerParserI interface <http://nltk.org/api/nltk.chunk.html#nltk.chunk.api.ChunkParserI>`_. But before you can use this chunker, you must have a `trained tagger <http://nltk-trainer.readthedocs.org/en/latest/train_tagger.html#using-a-trained-tagger>`. You first use the tagger to tag a sentence, and then use a chunker to parse the tagged sentence with the ``chunker.parse(sent)`` method::
	>>> chunker.parse(tagged_words)

``chunker.parse(tagged_words)`` will return a `Tree <http://nltk.org/api/nltk.html#nltk.tree.Tree>`_ whose `subtrees <http://nltk.org/api/nltk.html#nltk.tree.Tree.subtrees>`_ will be chunks, and whose `leaves <http://nltk.org/api/nltk.html#nltk.tree.Tree.leaves>`_ are the original tagged words.
