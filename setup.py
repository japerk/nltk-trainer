import os.path
from setuptools import setup

def open_file(fname):
	return open(os.path.join(os.path.dirname(__file__), fname))

setup(
	name='nltk-trainer',
	packages=['nltk_trainer'],
	version='0.9',
	description='Train NLTK objects with 0 code',
	long_description=open_file('README.rst').read(),
	license='Apache',
	author='Jacob Perkins',
	author_email='japerk@gmail.com',
	url='https://github.com/japerk/nltk-trainer',
	# TODO: download_url
	keywords=['nltk', 'nlp', 'nlproc'],
	scripts=(
		'analyze_chunked_corpus.py',
		'analyze_chunker_coverage.py',
		'analyze_classifier_coverage.py',
		'analyze_tagged_corpus.py',
		'analyze_tagger_coverage.py',
		'combine_classifiers.py',
		'train_chunker.py',
		'train_classifier.py',
		'train_tagger.py',
	),
	install_requires=[l.strip() for l in open_file('requirements.txt').readlines()],
	classifiers = [
		'Development Status :: 4 - Beta',
		'Environment :: Console',
		'Intended Audience :: Developers',
		'License :: OSI Approved :: Apache Software License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 2.7',
		'Topic :: Software Development :: Libraries :: Python Modules',
		'Topic :: Text Processing :: Linguistic',
	]
)
