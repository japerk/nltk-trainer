from setuptools import setup

setup(
	name='nltk-trainer',
	version='0.5',
	author='Jacob Perkins',
	author_email='japerk@gmail.com',
	url='https://github.com/japerk/nltk-trainer',
	packages=['nltk_trainer'],
	scripts=(
		'analyze_tagged_corpus.py', 'analyze_tagger_coverage.py',
		'train_chunker.py', 'train_classifier.py', 'train_tagger.py',
	),
	license='LICENSE',
	description='Train NLTK objects with 0 code',
	long_description=open('README.rst').read(),
	install_requires=(
		'argparse>=1.1',
		'nltk>=2.0b9',
		'numpy==1.3.0',
		'scipy==0.7.0',
	)
)