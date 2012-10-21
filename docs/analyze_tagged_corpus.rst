Analyzing a Tagged Corpus
-------------------------

The ``analyze_tagged_corpus.py`` script will show the following statistics about a tagged corpus:

 * total number of words
 * number of unique words
 * number of tags
 * the number of times each tag occurs

Example output can be found in `Analyzing Tagged Corpora and NLTK Part of Speech Taggers <http://streamhacker.com/2011/03/23/analyzing-tagged-corpora-nltk-part-speech-taggers/>`_.

To analyze the treebank corpus::
	``python analyze_tagged_corpus.py treebank``

To sort the output by tag count from highest to lowest::
	``python analyze_tagged_corpus.py treebank --sort count --reverse``

To see simplified tags, instead of standard tags::
	``python analyze_tagged_corpus.py treebank --simplify_tags``

To analyze a custom corpus, whose fileids end in ".pos", using a `TaggedCorpusReader <http://nltk.org/api/nltk.corpus.reader.html#nltk.corpus.reader.tagged.TaggedCorpusReader>`_::
	``python analyze_tagged_corpus.py /path/to/corpus --reader nltk.corpus.reader.tagged.TaggedCorpusReader --fileids '.+\.pos'``

The corpus path can be absolute, or relative to a nltk_data directory. For example, both ``corpora/treebank/tagged`` and ``/usr/share/nltk_data/corpora/treebank/tagged`` will work.

For a complete list of usage options::
	``python analyze_tagged_corpus.py --help``
