from nltk.tag import brill

def train_brill_tagger(initial_tagger, train_sents, end, trace=0, **kwargs):
	bounds = [(1, end)]
	
	templates = [
		brill.SymmetricProximateTokensTemplate(brill.ProximateTagsRule, *bounds),
		brill.SymmetricProximateTokensTemplate(brill.ProximateWordsRule, *bounds),
	]
	
	trainer = brill.FastBrillTaggerTrainer(initial_tagger, templates,
		deterministic=True, trace=trace)
	return trainer.train(train_sents, **kwargs)