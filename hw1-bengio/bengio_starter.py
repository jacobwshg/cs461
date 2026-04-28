
import argparse
import random

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

UNK = "<unk>"

def read_corpus( file_name, vocab, words, corpus, threshold ):
	#
	# `words` entry: { token: [ wID, count ] }
	#	
	wID = len( vocab )

	tokens = []
	with open( file_name, "rt" ) as f:
		for line in f:
			line = line.replace( "\n", "" )
			# parse raw tokens in line
			tokens.extend( line.split( " " ) )

	if threshold > -1:
		# count frequencies
		for t in tokens:
			try:
				elem = words[ t ]
			except:
				elem = [ wID, 0 ]
				vocab.append( t )
				wID = wID + 1
			elem[ 1 ] = elem[ 1 ] + 1
			words[ t ] = elem

		temp = words
		words = {}
		vocab = []
		wID = 0
		words[ UNK ] = [ wID, 100 ]
		vocab.append( UNK )
		for t in temp:
			# only append frequent words to vocabulary
			if temp[ t ][ 1 ] >= threshold:
				vocab.append( t )
				wID = wID + 1
				words[ t ] = [ wID, temp[ t ][ 1 ] ]

	for t in tokens:
		try:
			wID = words[ t ][ 0 ]
		except:
			wID = words[ UNK ][ 0 ]
		corpus.append( wID )

	return [ vocab, words, corpus ]

def encode( text, words ):
		encoded = []
		tokens = text.split( " " )
		for i in range( len( tokens ) ):
			try:
				wID = words[ tokens[ i ] ][ 0 ]
			except:
				wID = words[ UNK ][ 0 ]
			encoded.append( wID )
		return encoded

class bengio( torch.nn.Module ):
	def __init__( self, dim=50, window=3, batchsize=1, vocab_size=33279, activation=torch.tanh ):
		super().__init__()

		# specify weights, activation functions and any "helper" function needed for the neural net

		# activation
		self.activ_fn = activation

		# vocabulary size
		self.V = vocab_size
		# number of features
		self.m = dim
		# number of hidden units ( since only one `dim` provided,
		# use same value as # of features )
		self.h = dim
		# context size
		self.n = window
		# embeddings
		self.C = nn.Embedding( self.V, self.m )
		# hidden layer weights ( including biases d )
		self.H = nn.Linear( self.n * self.m, self.h )
		# output layer weights ( including biases b )
		self.U = nn.Linear( self.h, self.V )

	def forward( self, X_ ):
		# perform a forward pass ( inference ) on a batch of concatenated word embeddings
		# hint: it may be more efficient to pass a matrix of indices for the context, and
		# perform a look-up and concatenation of the word embeddings on the GPU.

		# assuming `X_` is word IDs in a given context, look up and concatenate embeddings
		X = self.C( X_ )
		X = X.view( X.size( 0 ), -1 )

		h = self.activ_fn( self.H( X ) )
		y_pred = self.U( h )

		#
		# |V| log probs, one for each vocabulary word ( including the true target
		# following this context in the corpus )
		#
		pred_ln_probs = torch.log_softmax( y_pred, dim=1 )
		return pred_ln_probs

def cross_entropy_loss( pred_ln_probs, y_true )
	sampl_cnt = pred_ln_probs.size( 0 )
	true_ln_probs = pred_ln_probs[ torch.arange( sampl_cnt ), y_true ]
	loss = -true_ln_probs.mean()
	return loss

def train( model, opt ):
	# implement code to split you corpus into batches, use a sliding window to construct contexts over
	# your batches ( sub-corpora ). you can manually replicate the functionality of dataloader() to present 
	# training examples to you model, you can manually calculate the probability assigned to the target 
	# token using torch matrix operations ( note: a mask to isolate the target word in the numerator may help ), 
	# calculate the negative average ln( prob ) over the batch and perform gradient descent.  you may want to loop
	# over the number of epochs internal to this function or externally.  it is helpful to report training
	# perplexity, percent complete and training speed as words-per-second.  it is also prudent to save
	# you model after every epoch.
	#
	# inputs to your neural network can be either word embeddings or word look-up indices

	n, batch_sz = opt.window, opt.batchsize
	batch_cnt = ( len( opt.train ) - n ) // batch_sz
	toks_per_batch = n * batch_sz # counting token overlaps between contexts

	LOG_INTERVAL = 1024
	starttime = time.time()
	tok_cnt = 0

	model.train()

	for i_batch in range( batch_cnt ):
		batch_base = i * batch_sz
		ctx_bases = [ ctx_base for ctx_base in range( batch_base, batch_base + batch_sz ) ]
		# contexts in batch; each element is a tensor of word IDs in a given context
		X = torch.stack(
			[\
				torch.tensor( opt.train[ ctx_base : ctx_base+n ] )\
				for ctx_base in ctx_bases\
			]
		)
		# targets in batch corresponding to each context
		y_true = torch.tensor(
			[ opt.train( [ ctx_base+n ] ) for ctx_base in ctx_bases ]
		)

		if not opt.no_cuda:
			X, y_true = X.cuda(), y_true.cuda()

		opt.optimizer.zero_grad()
		pred_ln_probs = model( X ) 

		loss = cross_entropy_loss( pred_ln_probs, y_true )
		loss.backward()
		opt.optimizer.step()

		tok_cnt += toks_per_batch

		if i % LOG_INTERVAL == 0:
			delta_time = time.time() - starttime
			wps = tok_cnt // delta_time
			ppl = torch.exp( loss ).item() # synch
			progress = ( i_batch / batch_cnt ) * 100

			print(
				f"batch { i_batch }/{ batch_cnt } ( { progress:.2f}% ), "
				f"loss { loss.item():4f}, "
				f"perplexity { ppl:.2f}, "
				f"speed: { wps } wps"
			)

			tok_cnt = 0
			starttime = time.time()

	if opt.savename:
		torch.save( model.state_dict(), opt.savename + "/model_weights" )
	return

def test_model( model, opt, epoch ):
	# functionality for this function is similar to train() except that you construct examples for the
	# test or validation corpus; and you do not apply gradient descent.
	return

def main():

	random.seed( 10 )

	parser = argparse.ArgumentParser()
	parser.add_argument( "-threshold", type=int, default=3 )
	parser.add_argument( "-window",    type=int, default=512 )   
	parser.add_argument( "-no_cuda",   action="store_true" )
	parser.add_argument( "-epochs",    type=int, default=20 )
	parser.add_argument( "-d_model",   type=int, default=512 )
	parser.add_argument( "-batchsize", type=int, default=1 )
	parser.add_argument( "-lr",        type=float, default=0.00001 )
	parser.add_argument( "-savename",  type=str )
	parser.add_argument( "-loadname",  type=str )

	opt = parser.parse_args()
	opt.verbose = False
	[ opt.vocab, opt.words, opt.train ] = read_corpus( "wiki2.train.txt", [], {}, [], opt.threshold )
	print( "vocab: %d train: %d" % ( len( opt.vocab ), len( opt.train ) ) )
	[ opt.vocab, opt.words, opt.test ] = read_corpus( "wiki2.test.txt", opt.vocab, opt.words, [], -1 )
	print( "vocab: %d test: %d" % ( len( opt.vocab ), len( opt.test ) ) )
	[ opt.vocab, opt.words, opt.valid ] = read_corpus( "wiki2.valid.txt", opt.vocab, opt.words, [], -1 )
	print( "vocab: %d test: %d" % ( len( opt.vocab ), len( opt.valid ) ) )

	print( "Train: %7d" % ( len( opt.train ) ) )
	print( "Test:  %7d" % ( len( opt.test ) ) )
	print( "Valid: %7d" % ( len( opt.valid ) ) )
	print( "Vocab: %7d" % ( len( opt.vocab ) ) )
	print( " " )

	opt.examples = []
	with open( "examples.txt", "rt" ) as f:
		for line in f:
			line = line.replace( "\n", "" )
			encoded = encode( line, opt.words )
			text = ""
			for i in range( len( encoded ) ):
				text = text + opt.vocab[ encoded[ i ] ] + " "
			opt.examples.append( encoded )

			print( "original: %s" % line )
			print( "encoded:  %s" % text )
			print( " " )

	dev = torch.device(
		"cuda" if torch.cuda.is_available else "cpu"
	)
	model = bengio( 
		dim=opt.d_model, 
		window=opt.window, 
		batchsize=opt.batchsize, 
		vocab_size=len( opt.vocab ), 
		activation=torch.tanh
	)
	if opt.no_cuda == False:
		model = model.cuda()
	opt.optimizer = torch.optim.Adam( model.parameters(), lr=opt.lr, betas=( 0.9, 0.98 ), eps=1e-9 )

	train( model, opt )
	test_model( model, opt, -1 )

if __name__ == "__main__":
	main()

