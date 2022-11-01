BertTopic
****************************

Introduction
------------------------
------------------------

BERTopic is a topic modeling technique that leverages BERT embeddings and a class-based TF-IDF to create dense clusters allowing for easily interpretable topics whilst keeping important words in the topic descriptions.

There are four key components used in BERTopic, those are:

	* A transformer embedding model
	* UMAP dimensionality reduction
	* HDBSCAN clustering
	* Cluster tagging using c-TF-IDF

.. image:: files/pics/BERTopic_overall_flowchart_2.png


Transformer Embedding
------------------------
------------------------

* The first step is to embed documents into dimensional vectors.
* BERTopic supports several libraries (Sentence Transformers, Flair, SpaCy, Gensim, USE TF Hub) for encoding our text to dense vector embeddings. Of these, `Sentence Transformers`_ library provides the most extensive library of high-performing sentence embedding models.
* As the name implies, this embedding model works best for either sentences or paragraphs. This means that whenever you have a set of documents, where each document contains several paragraphs, BERTopic will struggle to accurately extract a topic from that document. Several paragraphs typically means several topics and BERTopic will assign only one topic to a document. Therefore, it is advised to split up longer documents into either sentences or paragraphs before embedding them. That way, BERTopic will have a much easier job identifying topics in isolation.

* After building out embeddings, BERTopic compresses them into a lower-dimensional space. Thus, 384-dimensional vectors are transformed into two/three-dimensional vectors. To perform dimensionality reduction, we can use any of the popular choices such as `PCA`_, `tSNE`_, `UMAP`_, etc

UMAP
+++++++

+ Flexible non-linear dimension reduction algorithm
+ Learns the manifold structure of the data and find a low dimensional embedding that preserves the essential topological structure of that manifold

.. image:: files/pics/BERTopic_UMAP_3D_diagram_2.png


HDBSCAN Clustering
------------------------
------------------------

#. `HDBSCAN`_ is used to cluster the (now)low-dimensional vectors.

#. There are mainly two types of clustering methods:

	a) Flat or Hierarchical: Focuses on whether there is (or is not) a hierarchy in the clustering method. For example, we may (ideally) view our graph hierarchy as moving from continents to countries to cities.

	.. image:: files/pics/BERTopic_HDBSCAN_flat_or_hierarchical_2.png

	b) Centroid-based or Density-based: This means clustering based on proximity to a centroid or clustering based on the density of points. Centroid-based clustering is ideal for "spherical" clusters. Density-based clustering can handle more irregular shapes and identify outliers.

	.. image:: files/pics/BERTopic_Centroid_Density_cluster_2.png


#. HDBSCAN is a hierarchical, density-based method.
#. This means that we can benefit from the easier tuning and visualization of hierarchical data, handle irregular cluster shapes, and identify outliers.
#. HDBSCAN will identify and pick high-density regions, and eventually combine data points in these selected regions

	.. image:: files/pics/HDBSCAN_density_plot.png




.. _Sentence Transformers: https://www.pinecone.io/learn/sentence-embeddings/
.. _tSNE: https://medium.com/swlh/t-sne-explained-math-and-intuition-94599ab164cf
.. _PCA: https://towardsdatascience.com/principal-component-analysis-pca-explained-visually-with-zero-math-1cbf392b9e7d
.. _UMAP: https://pair-code.github.io/understanding-umap/
.. _HDBSCAN: https://pberba.github.io/stats/2020/07/08/intro-hdbscan/

Topic Extraction with c-TF-IDF
--------------------------------
--------------------------------

In the final step, BERTopic extracts topics for each of the clusters using a modified version on TF-IDF called c-TF-IDF.

`TF_IDF`_ is a popular technique for identifying the most relevant “documents” given a term or set of terms. c-TF-IDF turns this on its head by finding the most relevant terms given all of the “documents” within a cluster.

`Class Based TF_IDF`_: The goal of the class-based TF-IDF is to supply all documents within a single class with the same class vector. For this, we have to start looking at TF-IDF from a class-based point of view instead of individual documents. If documents are not individuals, but part of a larger collective, then it might be interesting to actually regard them as such by joining all documents in a class together.

.. _TF_IDF: https://medium.com/analytics-vidhya/tf-idf-term-frequency-technique-easiest-explanation-for-text-classification-in-nlp-with-code-8ca3912e58c3
.. _Class Based TF_IDF: https://maartengr.github.io/BERTopic/api/ctfidf.html

.. image:: files/pics/c_TF_IDF_Overview_Flowchart.png

Once we pick the most relevant terms for each cluster to derive topics, we can improve the coherence of words with `Maximal Marginal Relevance`_

.. _Maximal Marginal Relevance: https://maartengr.github.io/BERTopic/api/mmr.html



Model Execution
------------------------
------------------------


 Below is an overview of common functions in BERTopic:

.. image:: files/pics/BERTopic_default_params.png

 After having trained your BERTopic model, a number of attributes are saved within your model. These attributes, in part, refer to how model information is stored on an estimator during fitting. The attributes that you see below all end in _ and are public attributes that can be used to access model information.

.. image:: files/pics/BERTopic_Additional_params.png

Additional attributes can be found `here`_.

.. _here: https://maartengr.github.io/BERTopic/index.html#attributes



Listed below are the steps involved in executing the BERT out of box model.

1. Load the model

First step as usual, is to install the necessary BERT packages using PyPI.

.. code-block:: python

 		pip install bertopic

2. Model Fitting

The input documents will be loaded in as a list of strings. The steps are straightforward. Load in the dataset and preprocess if needed( Remove stop words and convert to list). For smaller datasets, it is preferable to remove stopwords.

.. code-block:: python

	from datasets import load_dataset
	from sklearn.feature_extraction.text import CountVectorizer
	data = load_dataset('jamescalam/reddit-python', split='train')
	# we add this to remove stopwords, for lower volumes of data stopwords can cause issues
	vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")
	# deal with df if needed
	if type(data['selftext']) is list:
		text = data['selftext']
	else:
	  text = data['selftext'].tolist()
	model = BERTopic(
	vectorizer_model=vectorizer_model,
	language='english', calculate_probabilities=True,
	verbose=True)
	topics, probs = model.fit_transform(text)


The outputs generated by fitting the model are the topics and probabilities. The topic value simply represents the topic it is assigned to and the probability is the likelihood of a document falling into any of the possible topics.

.. code-block:: python

		freq = model.get_topic_info()
		freq.head(10)

3. Visualize

Visualize Topics
This command shows the intertopic distance map between the topics that were generated in the BERTopic Model.

.. image:: files/pics/BERT_visual_topic.gif

.. code-block:: python

		model.visualize_topics()

Visualize Probabilities:

.. code-block:: python

		model.visualize_distribution(probabilities[0])

.. image:: files/pics/probability_BERT.png


You can also visualize the probabilities of documents belonging to a certain topic. Probabilities helps us understand how confident the model is for each instance.

4. Topic Reduction

Running Topic Models could generate hundreds of topics. This might sometimes be too much to explore and you may not require that level of granular knowledge. We use topic reduction to reduce the number of topics created. We can do this in 2 ways.

- Manual Topic Reduction:

 .. code-block:: python

 		from bertopic import BertTopic
		model = BERTopic(nr_topics=20)

 This forces the model to split topics into 20 different topics. You can do this when you already have an idea of how many topics could be generated from the documents.

- Automatic Topic Reduction:

 Most times, we are not aware of the number of topics that could ideally be generated from the documents. Automatic reduction reduces the number of topics iteratively as long as a pair of topics is found that exceeds a minimum similarity of 0.9.

- Topic Reduction after Training:

 This method is used when you want to reduce the number of topics after you have trained the model.

	.. code-block:: python

	 	from bertopic import BERTopic
	 	model = BERTopic()
	 	topics, probs = model.fit_transform(docs)
	 	# Further reduce topics
	 	new_topics, new_probs = model.reduce_topics(docs, topics, probs, nr_topics=30)

5. Topic Representation

By default, a class based TF-IDF is used to extract topics from the documents. If you want to change the way the words are represented in the topic, we can use the update_topics function to update the topic representation with new parameters.

.. code-block:: python

		# Update topic representation by increasing n-gram range and removing english stopwords
		model.update_topics(docs, topics, n_gram_range=(1, 3), stop_words="english")

Using a custom CountVectorizer

..code-block:: python

		from sklearn.feature_extraction.text import CountVectorizer
		cv = CountVectorizer(ngram_range=(1, 3), stop_words="english")
		model.update_topics(docs, topics, vectorizer=cv)

6. Custom Embedding
If you dont want to use pre-trained embedddings and created an embedding model, embed your documents with the model you trained and pass the embeddings to BERTopic.

..code-block:: python

		from bertopic import BERTopic
		from sentence_transformers import SentenceTransformer

		# Prepare embeddings
		sentence_model = SentenceTransformer("distilbert-base-nli-mean-tokens")
		embeddings = sentence_model.encode(docs, show_progress_bar=False)

		# Create topic model
		model = BERTopic()
		topics, probabilities = model.fit_transform(docs, embeddings)


Model Evaluation
------------------------
------------------------

**Coherence Score**

We can use coherence score to measure the performance of BERTopic model using coherence score. Coherence score is used to measure how interpretable the topics are to humans.

.. code-block:: python

	import gensim.corpora as corpora
	from gensim.models.coherencemodel import CoherenceModel
	# Preprocess documents
	cleaned_docs = topic_model._preprocess_text(docs)
	# Extract vectorizer and tokenizer from BERTopic
	vectorizer = topic_model.vectorizer_model
	tokenizer = vectorizer.build_tokenizer()
	# Extract features for Topic Coherence evaluation
	words = vectorizer.get_feature_names()
	tokens = [tokenizer(doc) for doc in cleaned_docs]
	dictionary = corpora.Dictionary(tokens)
	corpus = [dictionary.doc2bow(token) for token in tokens]
	topic_words = [[words for words, _ in topic_model.get_topic(topic)]
						 for topic in range(len(set(topics))-1)]
	# Evaluate
	coherence_model = CoherenceModel(topics=topic_words,
											 texts=tokens,
											 corpus=corpus,
											 dictionary=dictionary,
											 coherence='c_v')
	coherence = coherence_model.get_coherence()

**Using OCTIS**

OCTIS (Optimizing and Comparing Topic models Is Simple) aims at training, analyzing and comparing Topic Models, whose optimal hyperparameters are estimated by means of a Bayesian Optimization approach

The corresponding link to the associated Github page can be found `here`

.. _here: https://maartengr.github.io/BERTopic/index.html#attributes

OCTIS already supports the below models:

.. image:: files/pics/OCTIS_list.png

TEst 123
Since BERTopic is not implemented yet in the OCTIS module, we have to incorporate this model. Models inherit from the class AbstractModel defined in octis/models/model.py. To build your own model your class must override the train_model(self, dataset, hyperparameters) method which always requires at least a Dataset object and a Dictionary of hyperparameters as input and should return a dictionary with the output of the model as output.
