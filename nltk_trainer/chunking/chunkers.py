import itertools
import nltk.chunk
from nltk.tag import UnigramTagger, BigramTagger, ClassifierBasedTagger

#####################
## tree conversion ##
#####################

def chunk_trees2train_chunks(chunk_sents):
	tag_sents = [nltk.chunk.tree2conlltags(sent) for sent in chunk_sents]
	return [[((w,t),c) for (w,t,c) in sent] for sent in tag_sents]

def conll_tag_chunks(chunk_sents):
	'''Convert each chunked sentence to list of (tag, chunk_tag) tuples,
	so the final result is a list of lists of (tag, chunk_tag) tuples.
	>>> import nltk.chunk
	>>> from nltk.tree import Tree
	>>> t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])
	>>> conll_tag_chunks([t])
	[[('DT', 'B-NP'), ('NN', 'I-NP')]]
	'''
	tagged_sents = [nltk.chunk.tree2conlltags(tree) for tree in chunk_sents]
	return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

#################
## tag chunker ##
#################

class TagChunker(nltk.chunk.ChunkParserI):
	'''Chunks tagged tokens using Ngram Tagging.'''
	def __init__(self, train_chunks, tagger_classes=[UnigramTagger, BigramTagger]):
		'''Train Ngram taggers on chunked sentences'''
		train_sents = conll_tag_chunks(train_chunks)
		self.tagger = None
		
		for cls in tagger_classes:
			self.tagger = cls(train_sents, backoff=self.tagger)
	
	def parse(self, tagged_sent):
		'''Parsed tagged tokens into parse Tree of chunks'''
		if not tagged_sent: return None
		(words, tags) = zip(*tagged_sent)
		chunks = self.tagger.tag(tags)
		# create conll str for tree parsing
		wtc = itertools.izip(words, chunks)
		return nltk.chunk.conlltags2tree([(w,t,c) for (w,(t,c)) in wtc])

########################
## classifier chunker ##
########################

def prev_next_pos_iob(tokens, index, history):
	word, pos = tokens[index]
	
	if index == 0:
		prevword, prevpos, previob = ('<START>',)*3
	else:
		prevword, prevpos = tokens[index-1]
		previob = history[index-1]
	
	if index == len(tokens) - 1:
		nextword, nextpos = ('<END>',)*2
	else:
		nextword, nextpos = tokens[index+1]
	
	feats = {
		'word': word,
		'pos': pos,
		'nextword': nextword,
		'nextpos': nextpos,
		'prevword': prevword,
		'prevpos': prevpos,
		'previob': previob
	}
	
	return feats

class ClassifierChunker(nltk.chunk.ChunkParserI):
	def __init__(self, train_sents, feature_detector=prev_next_pos_iob, **kwargs):
		if not feature_detector:
			feature_detector = self.feature_detector
		
		train_chunks = chunk_trees2train_chunks(train_sents)
		self.tagger = ClassifierBasedTagger(train=train_chunks,
			feature_detector=feature_detector, **kwargs)
	
	def parse(self, tagged_sent):
		if not tagged_sent: return None
		chunks = self.tagger.tag(tagged_sent)
		return nltk.chunk.conlltags2tree([(w,t,c) for ((w,t),c) in chunks])