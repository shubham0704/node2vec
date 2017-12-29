'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.graph:
		G1 = nx.read_edgelist(nx.generate_edgelist(args.graph, data=True), nodetype=int, create_using=nx.DiGraph())
		G = args.graph
		#print G.nodes(), G1.nodes()
		print G.edges(), G1.edges()
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1
		return G
	elif args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks, args):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks] 
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
	#model.wv.save_word2vec_format(args.output)
	
	return model

def model_maker(args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	nx_G = read_graph(args)
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length, args.meta_paths)
	model = learn_embeddings(walks, args)
	return model
