from nltk.tag.sequential import SequentialBackoffTagger
from nltk.probability import FreqDist
from nltk.tag import ClassifierBasedPOSTagger, TaggerI, str2tuple
from nltk_trainer.featx import phonetics
from nltk_trainer.featx.metaphone import dm

class PhoneticClassifierBasedPOSTagger(ClassifierBasedPOSTagger):
	def __init__(self, double_metaphone=False, metaphone=False, soundex=False, nysiis=False, caverphone=False, *args, **kwargs):
		self.funs = {}
		
		if double_metaphone:
			self.funs['double-metaphone'] = lambda s: dm(unicode(s))
		
		if metaphone:
			self.funs['metaphone'] = phonetics.metaphone
		
		if soundex:
			self.funs['soundex'] = phonetics.soundex
		
		if nysiis:
			self.funs['nysiis'] = phonetics.nysiis
		
		if caverphone:
			self.funs['caverphone'] = phonetics.caverphone
		# for some reason don't get self.funs if this is done first, but works if done last
		ClassifierBasedPOSTagger.__init__(self, *args, **kwargs)
	
	def feature_detector(self, tokens, index, history):
		feats = ClassifierBasedPOSTagger.feature_detector(self, tokens, index, history)
		s = tokens[index]
		
		for key, fun in self.funs.iteritems():
			feats[key] = fun(s)
		
		return feats

class MaxVoteBackoffTagger(SequentialBackoffTagger):
	def __init__(self, *taggers):
		self._taggers = taggers
	
	def choose_tag(self, tokens, index, history):
		tags = FreqDist()
		
		for tagger in self._taggers:
			tags.inc(tagger.choose_tag(tokens, index, history))
		
		return tags.max()

class PatternTagger(TaggerI):
	def tag(self, tokens):
		# don't import at top since don't want to fail if not installed
		from pattern.en import tag
		# not tokenizing ensures that the number of tagged tokens returned is
		# the same as the number of input tokens
		return tag(u' '.join(tokens), tokenize=False)