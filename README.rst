NLTK Trainer
------------

NLTK Trainer exists to make training and evaluating NLTK objects as easy as possible.

Requirements
------------

The scripts with default arguments have been tested for compatibility with Python3.7 and NLTK 3.4.5. If something does not work for you, please `open an issue <https://github.com/japerk/nltk-trainer/issues/new>`_. Include the script with arguments and failure or exception output. To use the sklearn classifiers, you must also install `scikit-learn <http://scikit-learn.org/stable/>`_.

If you want to use any of the corpora that come with NLTK, you should `install the NLTK data <http://nltk.org/data.html>`_.

Documentation
-------------

Documentation can be found at `nltk-trainer.readthedocs.org <http://nltk-trainer.readthedocs.org/en/latest/>`_ (you can also find these documents in the `docs directory <https://github.com/japerk/nltk-trainer/tree/master/docs>`_. Many of the scripts are covered in `Python 3 Text Processing with NLTK 3 Cookbook <http://www.amazon.com/gp/product/1782167854/ref=as_li_tl?ie=UTF8&camp=1789&creative=390957&creativeASIN=1782167854&linkCode=as2&tag=streamhacker-20&linkId=K2BYHHUBZ4GIEW4L>`_, and every script provides a ``--help`` option that describes all available parameters.

Using Trained Models
--------------------

The trained models are pickle files that by default are put into your ``nltk_data`` directory. You can load them using ``nltk.data.load``, for example::

    import nltk.data
    classifier = nltk.data.load('classifiers/movie_reviews_NaiveBayes.pickle')

You now have a NLTK classifier object you can work with.