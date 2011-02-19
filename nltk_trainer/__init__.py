import os, os.path
import cPickle as pickle

def dump_object(obj, fname, trace=1):
	dirname = os.path.dirname(fname)
	
	if not os.path.exists(dirname):
		if trace:
			print 'creating directory %s' % dirname
		
		os.mkdir(dirname)
	
	if trace:
		print 'dumping %s to %s' % (obj.__class__.__name__, fname)
	
	f = open(fname, 'wb')
	pickle.dump(obj, f)
	f.close()