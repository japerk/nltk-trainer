Training IOB Chunkers
---------------------

The ``train_chunker.py`` script can use any corpus included with NLTK that implements a ``chunked_sents()`` method.

Train the default sequential backoff tagger based chunker on the treebank_chunk corpus::
	``python train_chunker.py treebank_chunk``

To train a NaiveBayes classifier based chunker::
	``python train_chunker.py treebank_chunk --classifier NaiveBayes``

To train on the conll2000 corpus::
	``python train_chunker.py conll2000``

To train on a custom corpus, whose fileids end in ".pos", using a `ChunkedCorpusReader <http://nltk.googlecode.com/svn/trunk/doc/api/nltk.corpus.reader.chunked.ChunkedCorpusReader-class.html>`_::
	``python train_chunker.py /path/to/corpus --reader nltk.corpus.reader.chunked.ChunkedCorpusReader --fileids '.+\.pos'``

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

You can also restrict the files used with the ``--fileids`` option::
	``python train_chunker.py conll2000 --fileids train.txt``

For a complete list of usage options::
	``python train_chunker.py --help``
