import itertools
import nltk.tag
from nltk.chunk import ChunkParserI
from nltk.chunk.util import conlltags2tree, tree2conlltags
from nltk.tag import UnigramTagger, BigramTagger, ClassifierBasedTagger

#####################
## tree conversion ##
#####################

def chunk_trees2train_chunks(chunk_sents):
	tag_sents = [tree2conlltags(sent) for sent in chunk_sents]
	return [[((w,t),c) for (w,t,c) in sent] for sent in tag_sents]

def conll_tag_chunks(chunk_sents):
	'''Convert each chunked sentence to list of (tag, chunk_tag) tuples,
	so the final result is a list of lists of (tag, chunk_tag) tuples.
	>>> from nltk.tree import Tree
	>>> t = Tree('S', [Tree('NP', [('the', 'DT'), ('book', 'NN')])])
	>>> conll_tag_chunks([t])
	[[('DT', 'B-NP'), ('NN', 'I-NP')]]
	'''
	tagged_sents = [tree2conlltags(tree) for tree in chunk_sents]
	return [[(t, c) for (w, t, c) in sent] for sent in tagged_sents]

def ieertree2conlltags(tree, tag=nltk.tag.pos_tag):
	# tree.pos() flattens the tree and produces [(word, node)] where node is
	# from the word's parent tree node. words in a chunk therefore get the
	# chunk tag, while words outside a chunk get the same tag as the tree's
	# top node
	words, ents = zip(*tree.pos())
	iobs = []
	prev = None
	# construct iob tags from entity names
	for ent in ents:
		# any entity that is the same as the tree's top node is outside a chunk
		if ent == tree.node:
			iobs.append('O')
			prev = None
		# have a previous entity that is equal so this is inside the chunk
		elif prev == ent:
			iobs.append('I-%s' % ent)
		# no previous equal entity in the sequence, so this is the beginning of
		# an entity chunk
		else:
			iobs.append('B-%s' % ent)
			prev = ent
	# get tags for each word, then construct 3-tuple for conll tags
	words, tags = zip(*tag(words))
	return itertools.izip(words, tags, iobs)

#################
## tag chunker ##
#################

class TagChunker(ChunkParserI):
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
		return conlltags2tree([(w,t,c) for (w,(t,c)) in wtc])

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

class ClassifierChunker(ChunkParserI):
	def __init__(self, train_sents, feature_detector=prev_next_pos_iob, **kwargs):
		if not feature_detector:
			feature_detector = self.feature_detector
		
		train_chunks = chunk_trees2train_chunks(train_sents)
		self.tagger = ClassifierBasedTagger(train=train_chunks,
			feature_detector=feature_detector, **kwargs)
	
	def parse(self, tagged_sent):
		if not tagged_sent: return None
		chunks = self.tagger.tag(tagged_sent)
		return conlltags2tree([(w,t,c) for ((w,t),c) in chunks])

#############
## pattern ##
#############

class PatternChunker(ChunkParserI):
	def parse(self, tagged_sent):
		# don't import at top since don't want to fail if not installed
		from pattern.en import parse
		s = ' '.join([word for word, tag in tagged_sent])
		# not tokenizing ensures that the number of tagged tokens returned is
		# the same as the number of input tokens
		sents = parse(s, tokenize=False).split()
		if not sents: return None
		return conlltags2tree([(w, t, c) for w, t, c, p in sents[0]])
