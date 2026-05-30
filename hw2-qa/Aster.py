
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from torch.utils.data import Dataset, DataLoader
import torch.cuda.amp as amp
from tqdm import tqdm

import random

class GPT2Attention( nn.Module ):
	def __init__( self, d_model, heads, max_seq_len, attn_dropout=0.0, resid_dropout=0.0 ):
		super().__init__()

		assert d_model % heads == 0

		self.heads = heads
		self.d_model = d_model
		self.head_dim = d_model // heads

		self.c_attn = nn.Linear( d_model, 3 * d_model )
		self.c_proj = nn.Linear( d_model, d_model )

		self.attn_dropout = nn.Dropout( attn_dropout )
		self.resid_dropout = nn.Dropout( resid_dropout )

		bias = torch.tril( torch.ones( max_seq_len, max_seq_len, dtype=torch.bool ) ).view( 1, 1, max_seq_len, max_seq_len )
		self.register_buffer( "bias", bias, persistent=False )

	def forward( self, x, attention_mask=None ):
		bsz, seq_len, _ = x.size()

		qkv = self.c_attn( x )
		q, k, v = qkv.split( self.d_model, dim=2 )

		q = q.view( bsz, seq_len, self.heads, self.head_dim ).transpose( 1, 2 )
		k = k.view( bsz, seq_len, self.heads, self.head_dim ).transpose( 1, 2 )
		v = v.view( bsz, seq_len, self.heads, self.head_dim ).transpose( 1, 2 )

		scores = torch.matmul( q, k.transpose( -2, -1 ) ) / math.sqrt( self.head_dim )

		causal_mask = self.bias[ :, :, :seq_len, :seq_len ]
		scores = scores.masked_fill( ~causal_mask, torch.finfo( scores.dtype ).min )

		if attention_mask is not None:
			key_mask = attention_mask[ :, None, None, : ].to( torch.bool )
			scores = scores.masked_fill( ~key_mask, torch.finfo( scores.dtype ).min )

		probs = F.softmax( scores, dim=-1 )
		probs = self.attn_dropout( probs )

		attn = torch.matmul( probs, v )
		attn = attn.transpose( 1, 2 ).contiguous().view( bsz, seq_len, self.d_model )

		return self.resid_dropout( self.c_proj( attn ) )

# feed-forward
class GPT2MLP( nn.Module ):
	def __init__( self, d_model, d_ff, resid_dropout=0.0 ):
		super().__init__()

		self.c_fc = nn.Linear( d_model, d_ff )
		self.c_proj = nn.Linear( d_ff, d_model )

		try:
			self.act = nn.GELU( approximate="tanh" )
		except TypeError:
			self.act = nn.GELU()

		self.dropout = nn.Dropout( resid_dropout )

	def forward( self, x ):
		return self.dropout( self.c_proj( self.act( self.c_fc( x ) ) ) )

class GPT2Block( nn.Module ):
	def __init__( self, d_model, heads, d_ff, max_seq_len, attn_dropout=0.0, resid_dropout=0.0, layer_norm_epsilon=1e-5 ):
		super().__init__()

		self.ln_1 = nn.LayerNorm( d_model, eps=layer_norm_epsilon )
		self.attn = GPT2Attention( d_model, heads, max_seq_len, attn_dropout=attn_dropout, resid_dropout=resid_dropout )

		self.ln_2 = nn.LayerNorm( d_model, eps=layer_norm_epsilon )
		self.mlp = GPT2MLP( d_model, d_ff, resid_dropout=resid_dropout )

	def forward( self, x, attention_mask=None ):
		x = x + self.attn( self.ln_1( x ), attention_mask=attention_mask )
		x = x + self.mlp( self.ln_2( x ) )
		return x

class TransformerGPT( nn.Module ):
	def __init__( self, vocab_size, d_model, n_layers, heads, seqlen, d_ff, dropout=0.0, layer_norm_epsilon=1e-5 ):
		super().__init__()

		self.seqlen = seqlen

		# word token embedding
		self.wte = nn.Embedding( vocab_size, d_model )
		# word position embedding
		self.wpe = nn.Embedding( seqlen, d_model )

		self.drop = nn.Dropout( dropout )

		# N x GPT2Block
		self.h = nn.ModuleList( 
			[ 
				GPT2Block( 
					d_model, heads, 
					d_ff, seqlen, 
					attn_dropout=dropout, resid_dropout=dropout, 
					layer_norm_epsilon=layer_norm_epsilon
				) for _ in range( n_layers )
			]
		)

		self.ln_f = nn.LayerNorm( d_model, eps=layer_norm_epsilon )

		self.lm_head = nn.Linear( d_model, vocab_size, bias=False )
		self.lm_head.weight = self.wte.weight

	def forward( self, input_ids, attention_mask=None ):
		bsz, seq_len = input_ids.shape

		if seq_len > self.seqlen:
			raise ValueError( f"sequence length { seq_len } exceeds model seqlen { self.seqlen }" )

		pos = torch.arange( 0, seq_len, device=input_ids.device, dtype=torch.long ).unsqueeze( 0 )

		x = self.wte( input_ids ) + self.wpe( pos )
		x = self.drop( x )

		for block in self.h:
			x = block( x, attention_mask=attention_mask )

		x = self.ln_f( x )
		logits = self.lm_head( x )

		return x, logits


def load_model_best_state_dict( path ):
	checkpoint = torch.load( path, map_location="cpu" )

	if isinstance( checkpoint, dict ) and "model_state_dict" in checkpoint:
		state_dict = checkpoint[ "model_state_dict" ]
	elif isinstance( checkpoint, dict ) and "model" in checkpoint:
		state_dict = checkpoint[ "model" ]
	else:
		state_dict = checkpoint

	return state_dict

def my_tokenizer( path, tokenizer, max_tokens ):
	indices = []
	line_batch = []

	with open( path, "r", encoding="utf-8", errors="replace", newline="" ) as f:
		for line in f:
			encoded = tokenizer( line, add_special_tokens=False )[ "input_ids" ]
			if len( indices ) < max_tokens:
				for ids in encoded:
					indices.append( int( ids ) )
	return indices

@torch.no_grad()
def test_model( model, indices, opt, epoch=0, device=None ):
	start_time = time.time()

	if device is None:
		device = next( model.parameters() ).device

	aa = opt.seqlen
	bb = opt.batchsize
	total_loss = 0.0
	count = 0

	n_tokens = len( indices )
	vocab_size = model.wte.weight.size( 0 )
	stride = aa * bb

	with torch.no_grad():
		# drag ctx window
		for i in range( 0, n_tokens - aa + 1, stride ):
			src = torch.zeros( ( bb, aa ), dtype=torch.long )
			trg = torch.zeros( ( bb, aa - 1, vocab_size ), dtype=torch.float )
			actual_batchsize = 0
			for k in range( bb ):
				start_idx = i + k * aa
				if start_idx + aa > n_tokens:
					break
				for j in range( aa-1 ):
					src[ k, j ] = indices[ start_idx + j ]
					nxt_tok_id = indices[ start_idx + j + 1 ]
					trg[ k, j, indices[ start_idx + j + 1 ] ] = 1.0

				actual_batchsize += 1

			src = src[ :actual_batchsize ].to( device )
			trg = trg[ :actual_batchsize ].to( device )

			attention_mask = torch.ones_like( src, dtype=torch.long, device=device )

			x,preds = model( src, attention_mask=attention_mask )

			preds = preds[ :, :-1, : ]

			max_preds = torch.amax( preds, dim=2 ).unsqueeze( 2 )
			preds = preds - max_preds
			logits = torch.exp( preds )
			denoms = torch.sum( logits, 2 )
			denoms = denoms.unsqueeze( 2 )
			numer = logits * trg
			numer = torch.sum( numer, 2 )
			numer = numer.unsqueeze( 2 )
			probs = numer / denoms
			loss = -torch.log( probs + 1e-12 ).mean()
			print( i,loss.item() )

			total_loss += loss.item()
			count += 1

	avg_loss = total_loss / count
	ppl = math.exp( min( avg_loss, 20.0 ) )

	elapsed_min = int( ( time.time() - start_time ) // 60 )

	print( " " )
	print( "%dm: TEST %d [ %s ]  100%%  loss = %.3f" % ( elapsed_min, epoch + 1, "#" * 20, avg_loss ) )
	print( "epoch %d complete, loss = %.03f ppl = %7.1f" % ( epoch + 1, avg_loss, ppl ) )
	print( " " )

	return ppl

def read_obqa( file_name ):
	data = []
	with open( file_name,"rt" ) as f:
		for line in f:
			line = line.replace( "\n","" )
			tokens = line.split( "|" )
			d = {}
			d[ "fact" ] = tokens[ 0 ]
			d[ "stem" ] = tokens[ 1 ]
			d[ "A" ] = tokens[ 2 ]
			d[ "B" ] = tokens[ 3 ]
			d[ "C" ] = tokens[ 4 ]
			d[ "D" ] = tokens[ 5 ]
			d[ "Answer" ] = tokens[ 6 ]
			data.append( d )
	#for i in range( 5 ):
	#	print( i,data[ i ] )
	print( "data: %d" % ( len( data ) ) )
	return( data )

CHOICE_TAG_IDX_TBL = { "A":0, "B":1, "C":2, "D":3 }
ID_IGNORE = -100

#
# for classification
#
class OBQADataset( Dataset ):
	def __init__( self, data_list, tokenizer, max_len=128 ):

		self.data = data_list
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__( self ):
		return len( self.data )

	def __getitem__( self, idx ):
		item = self.data[ idx ]
		fact = item[ "fact" ].strip()
		stem = item[ "stem" ].strip()
		label_idx = CHOICE_TAG_IDX_TBL[ item[ "Answer" ].strip() ]

		choices = [ item[ "A" ], item[ "B" ], item[ "C" ], item[ "D" ] ]

		input_ids = []
		labels	= []
		attn_msks = []

		ID_PAD	= self.tokenizer.pad_token_id

		for choice in choices:

			ctx_seq = f"Fact: { fact } Question: { stem } Answer:"
			trg_seq = f" { choice.strip() } "

			ctx_ids = self.tokenizer.encode( ctx_seq, add_special_tokens=False )
			trg_ids = self.tokenizer.encode( trg_seq, add_special_tokens=False )
			ctx_len = len( ctx_ids )
			input_len = ctx_len + len( trg_ids )

			choice_input_ids = [ ID_PAD ] * self.max_len
			choice_labels = [ ID_IGNORE ] * self.max_len
			choice_attnmsk = [ 0 ] * self.max_len

			"""
			for i in range( self.max_len ):

				if i < ctx_len:
					# ID is in ctx
					choice_input_ids[ i ] = ctx_ids[ i ]

				elif i < input_len:
					# ID is in target
					choice_input_ids[ i ] = trg_ids[ i-ctx_len ]
					# only calculate loss on target tokens; mask out ctx tokens
					choice_labels[ i ] = choice_input_ids[ i ]

				if i < input_len:
					choice_attnmsk[ i ] = 1
			"""

			for i in range( min( self.max_len, input_len ) ):
				choice_attnmsk[ i ] = 1

				if i < ctx_len:
					# ID is in ctx
					choice_input_ids[ i ] = ctx_ids[ i ]
				else:
					# ID is in target
					choice_input_ids[ i ] = trg_ids[ i-ctx_len ]
					# only calculate loss on target tokens; mask out label for ctx tokens
					choice_labels[ i ] = choice_input_ids[ i ]

			input_ids.append( choice_input_ids )
			labels   .append( choice_labels )
			attn_msks.append( choice_attnmsk )

		return \
		{ 
			# ( 4, max_len )
			"input_ids"	 : torch.tensor( input_ids, dtype=torch.long ),
			"attention_mask": torch.tensor( attn_msks, dtype=torch.long ),
			"labels"		: torch.tensor( labels,	dtype=torch.long ),
			"label_idx"	 : torch.tensor( label_idx,  dtype=torch.long )
		}

def forward_on_qa_batch( model, batch, dev="cuda" ):
	input_ids = batch[ "input_ids" ].to( dev )
	attn_msk  = batch[ "attention_mask" ].to( dev )
	labels	  = batch[ "labels" ].to( dev )
	label_idxs = batch[ "label_idx" ].to( dev ) # ( batch_size )

	batch_sz, num_choices, seq_len = input_ids.shape

	input_ids_flat = input_ids.view( -1, seq_len )
	attn_msk_flat = attn_msk.view( -1, seq_len )
	labels_flat	= labels.view( -1, seq_len )

	_x, y = model( input_ids_flat, attention_mask=attn_msk_flat )

	cur_logits = y[ ..., :-1, : ].contiguous()
	nxt_labels = labels_flat[ ..., 1: ].contiguous()

	loss_fn = nn.CrossEntropyLoss( reduction="none" )
	loss = loss_fn( 
		cur_logits.view( -1, cur_logits.size( -1 ) ),
		nxt_labels.view( -1 )
	)
	#
	# ignore loss on final seq token, because we only used its label
	# and didn't use its logit
	#
	loss_flat = loss.view( batch_sz * num_choices, seq_len-1 )

	valid_tok_cnts = ( nxt_labels != ID_IGNORE ).sum( dim=-1 ).float()
	valid_tok_cnts = torch.clamp( valid_tok_cnts, min=1.0 )

	seq_losses = loss_flat.sum( dim=-1 ) / valid_tok_cnts
	choice_losses = seq_losses.view( batch_sz, num_choices )

	# predict choice with lowest loss
	preds = torch.argmin( choice_losses, dim=-1 )

	return choice_losses, preds, batch_sz

scaler = amp.GradScaler()

def train_mcqa( model, dataloader, optimizer, dev ):
	model.train()

	train_loss, train_acc = 0.0, 0.0
	total_loss = 0
	total, correct = 0, 0

	progbar = tqdm( dataloader, desc="training decoder-only MCQA" )

	for batch in progbar:

		optimizer.zero_grad()

		#
		# use mixed precision to save memory and improve speed
		#
		with amp.autocast( 
			device_type="cuda" if "cuda" in str( dev ) else "cpu",	
			dtype=torch.float16
		):
			choice_losses, preds, batch_sz = forward_on_qa_batch( 
				model, batch,
				dev
			)

			label_idxs = batch[ "label_idx" ].to( dev )

			# now lower loss = higher score
			clsn_loss = nn.CrossEntropyLoss()( 
				-choice_losses,
				label_idxs
			)

		scaler.scale( clsn_loss ).backward()
		scaler.step( optimizer )
		scaler.update()

		total_loss += clsn_loss.item() * batch_sz
		total += batch_sz
		correct += ( preds == label_idxs ).sum().item()

		train_loss = total_loss / total
		train_acc  = correct / total
		progbar.set_postfix( 
			{ 
				"loss": train_loss,
				"acc" : train_acc
			}
		)

	return train_loss, train_acc

@torch.no_grad()
def eval_mcqa( model, dataloader, dev ):

	model.eval()

	correct, total = 0, 0

	progbar = tqdm( dataloader, desc="evaluating decoder-only MCQA" )
	for batch in progbar:
		_choice_losses, preds, batch_sz = forward_on_qa_batch( 
			model, batch,
			dev
		)

		total += batch_sz 
		correct += ( preds == batch[ "label_idx" ].to( dev ) ).sum().item()

	eval_acc = correct / total
	return eval_acc

class OBQAGenerationDataset( Dataset ):
	def __init__( self, data_list, tokenizer, max_ctx_len=96, max_choice_len=32 ):
		self.data = data_list
		self.tokenizer = tokenizer
		self.max_ctx_len = max_ctx_len
		self.max_choice_len = max_choice_len

	def __len__( self ):
		return len( self.data )

	def __getitem__( self, idx ):
		item = self.data[ idx ]
		fact = item[ "fact" ].strip()
		stem = item[ "stem" ].strip()
		label_idx = CHOICE_TAG_IDX_TBL[ item[ "Answer" ].strip() ]
		choices = [ item[ "A" ], item[ "B" ], item[ "C" ], item[ "D" ] ]

		# format and tokenize generation context prefix
		ctx_seq = f"Fact: { fact } Question: { stem } Answer:"
		ctx_ids = self.tokenizer.encode( ctx_seq, add_special_tokens=False )
		ctx_ids = ctx_ids[ :self.max_ctx_len ]  # Truncate if it exceeds bounds

		# tokenize and pad each choice target separately for BERTScore matrix operations
		choice_ids_list = []
		for choice in choices:
			c_ids = self.tokenizer.encode( f" { choice.strip() } ", add_special_tokens=False )
			c_ids = c_ids[ :self.max_choice_len ]
			
			# pad choices to fixed length so they can be stacked into a single tensor
			padded_c_ids = c_ids + [ self.tokenizer.pad_token_id ] * ( self.max_choice_len - len( c_ids ) )
			choice_ids_list.append( padded_c_ids )

		return {
			"context_input_ids": torch.tensor( ctx_ids, dtype=torch.long ),
			"choice_ids": torch.tensor( choice_ids_list, dtype=torch.long ), # Shape: ( 4, max_choice_len )
			"label_idx": torch.tensor( label_idx, dtype=torch.long )
		}


def beam_search( model, context_ids, beam_width=3, max_gen_len=32, temperature=1.0, pad_token_id=0 ):
	"""
	perform token-by-token beam search tracking cumulative log probs
	"""
	model.eval()
	device = context_ids.device

	# each beam: { "tokens": 1D tensor, "log_prob": float }
	beams = [ { "tokens": context_ids.clone(), "log_prob": 0.0 } ]

	for _ in range( max_gen_len ):
		candidates = []

		for beam in beams:
			tokens = beam[ "tokens" ]
			
			# avoid predicting past the maximum allowed len
			if tokens.size( 0 ) >= model.seqlen:
				candidates.append( beam )
				continue

			# forward pass to retrieve logits for final token
			with torch.no_grad():
				_, logits = model( tokens.unsqueeze( 0 ) )
				next_token_logits = logits[ 0, -1, : ] / max( temperature, 1e-5 )
				log_probs = F.log_softmax( next_token_logits, dim=-1 )

			# select top K candidates for expansion
			topk_log_probs, topk_ids = torch.topk( log_probs, beam_width )

			for i in range( beam_width ):
				next_tok = topk_ids[ i ].unsqueeze( 0 )
				cand_tokens = torch.cat( [ tokens, next_tok ], dim=0 )
				cand_log_prob = beam[ "log_prob" ] + topk_log_probs[ i ].item()
				candidates.append( { "tokens": cand_tokens, "log_prob": cand_log_prob } )

		# sort candidates based on cumulative log prob and retain top beams
		candidates = sorted( candidates, key=lambda x: x[ "log_prob" ], reverse=True )
		beams = candidates[ :beam_width ]

		# TODO early stopping check if all top beams have finished

	# return the top-scoring sequence, excluding context prefix
	best_tokens = beams[ 0 ][ "tokens" ]
	return best_tokens[ len( context_ids ): ]

def compute_bertscore_wte( model, gen_ids, ref_ids ):
	"""
	compute static embedding-based BERTScore metric
	"""
	if len( gen_ids ) == 0 or len( ref_ids ) == 0:
		return torch.tensor( 0.0, device=gen_ids.device )

	# extract static token embeddings from the model
	gen_emb = model.wte( gen_ids ).float()  # Shape: [ M, d_model ]
	ref_emb = model.wte( ref_ids ).float()  # Shape: [ N, d_model ]

	# normalize token embeddings to calculate cosine similarity
	gen_emb = gen_emb / ( gen_emb.norm( dim=-1, keepdim=True ) + 1e-8 )
	ref_emb = ref_emb / ( ref_emb.norm( dim=-1, keepdim=True ) + 1e-8 )
	# construct similarity matrix across token sequences
	sim_matrix = torch.matmul( gen_emb, ref_emb.T )  # Shape: [ M, N ]

	# calculate precision, recall, and f1 score
	precision = sim_matrix.max( dim=1 )[ 0 ].mean()
	recall = sim_matrix.max( dim=0 )[ 0 ].mean()

	f1 = 2 * ( precision * recall ) / ( precision + recall + 1e-8 )
	return f1

def get_sequence_log_prob( model, context_ids, gen_ids ):
	"""
	re-run forward pass with gradient graph tracking 
	to compute differentiable log probs of generated tokens
	"""
	full_seq = torch.cat( [ context_ids, gen_ids ], dim=0 )
	_, logits = model( full_seq.unsqueeze( 0 ) )

	# align logits with the targeted generation targets
	ctx_len = len( context_ids )
	gen_logits = logits[ 0, ctx_len - 1 : -1, : ]

	log_probs = F.log_softmax( gen_logits, dim=-1 )
	target_log_probs = log_probs.gather( 1, gen_ids.unsqueeze( 1 ) ).squeeze( 1 )

	return target_log_probs.sum()


def train_generative( model, dataloader, optimizer, dev, beam_width=3, temperature=0.7 ):
	model.train()
	total_loss = 0.0
	total_samples = 0
	correct_mappings = 0

	progbar = tqdm( dataloader, desc="training beam search" )

	for batch in progbar:
		optimizer.zero_grad()

		# squeeze out the batch dimension since batch_size=1
		ctx_ids = batch[ "context_input_ids" ][ 0 ].to( dev ) # ( ctx_len )
		choice_ids = batch[ "choice_ids" ][ 0 ].to( dev )     # ( 4, max_choice_len )
		gt_label_idx = batch[ "label_idx" ][ 0 ].item()       # scalar

		# generation pass
		with torch.no_grad():
			generated_tokens = beam_search( 
				model=model, 
				context_ids=ctx_ids, 
				beam_width=beam_width, 
				temperature=temperature
			)

		if len( generated_tokens ) == 0:
			continue

		# compute BERTScore against target labels
		scores = []
		for i in range( 4 ):
			# unpad choice tokens before passing to BERTScore calculation
			valid_choice_tokens = choice_ids[ i ][ choice_ids[ i ] != dataloader.dataset.tokenizer.pad_token_id ]
			score = compute_bertscore_wte( model, generated_tokens, valid_choice_tokens )
			scores.append( score )

		scores_tensor = torch.stack( scores )
		predicted_choice_idx = torch.argmax( scores_tensor ).item()

		total_samples += 1
		if predicted_choice_idx == gt_label_idx:
			correct_mappings += 1

		# formulate reward
		gt_score = scores_tensor[ gt_label_idx ]
		mask = torch.ones( 4, dtype=torch.bool, device=dev )
		mask[ gt_label_idx ] = False
		max_wrong_score = scores_tensor[ mask ].max()

		reward = gt_score - max_wrong_score

		# minimize NLL scaled by reward performance
		log_prob_seq = get_sequence_log_prob( model, ctx_ids, generated_tokens )
		loss = -log_prob_seq * reward

		loss.backward()
		optimizer.step()

		total_loss += loss.item()

		progbar.set_postfix( 
			{ 
				"loss": loss.item(),
				"acc" : correct_mappings / total_samples
			}
		)


	return total_loss / len( dataloader ), correct_mappings / len( dataloader )

@torch.no_grad()
def eval_generative( 
	model, 
	data_list, 
	tokenizer, 
	dev, 
	num_samples_to_print=3, 
	beam_width=3, 
	temperature=0.7
):
	"""
	evaluate the model using the beam search + BERTScore
	randomly select and log details for a few samples
	"""
	model.eval()
	
	# randomly select sample indices to log
	total_samples = len( data_list )
	samples_to_log = set( random.sample( range( total_samples ), min( num_samples_to_print, total_samples ) ) )

	correct_mappings = 0
	idx_to_tag = { 0: "A", 1: "B", 2: "C", 3: "D" }

	print( f"start evaluation over { total_samples } samples. \ndetailed logs will print for indices: { list( samples_to_log ) }\n" )

	for idx in range( total_samples ):
		# extract original raw data tokens and dictionary attributes
		item = data_list[ idx ]
		fact = item[ "fact" ].strip()
		stem = item[ "stem" ].strip()
		gt_tag = item[ "Answer" ].strip()
		gt_label_idx = CHOICE_TAG_IDX_TBL[ gt_tag ]
		choices = [ item[ "A" ].strip(), item[ "B" ].strip(), item[ "C" ].strip(), item[ "D" ].strip() ]

		# format ctx prefix
		ctx_seq = f"Fact: { fact } Question: { stem } Answer:"
		ctx_ids = torch.tensor( tokenizer.encode( ctx_seq, add_special_tokens=False ), dtype=torch.long, device=dev )

		# tokenize target choice sequences
		choice_token_lists = [ 
			torch.tensor( tokenizer.encode( f" { c } ", add_special_tokens=False ), dtype=torch.long, device=dev )
			for c in choices
		]

		# run beams search
		generated_tokens = beam_search( 
			model=model, 
			context_ids=ctx_ids, 
			beam_width=beam_width, 
			temperature=temperature,
			pad_token_id=tokenizer.pad_token_id
		)

		# compute BERTScores across all 4 choices
		scores_tensor = torch.zeros( 4, device=dev )
		if len( generated_tokens ) > 0:
			scores = []
			for choice_tokens in choice_token_lists:
				score = compute_bertscore_wte( model, generated_tokens, choice_tokens )
				scores.append( score )
			scores_tensor = torch.stack( scores )
			predicted_choice_idx = torch.argmax( scores_tensor ).item()
		else:
			# fallback guess if empty output
			predicted_choice_idx = 0 

		if predicted_choice_idx == gt_label_idx:
			correct_mappings += 1

		# log selected samples
		if idx in samples_to_log:
			# decode token sequence back to text
			gen_text = tokenizer.decode( generated_tokens, skip_special_tokens=True ).strip()
			pred_tag = idx_to_tag[ predicted_choice_idx ]

			##print( "=" * 70 )
			print()
			print( f" [ EVAL DIAGNOSTICS ] SAMPLE INDEX: { idx } " )
			##print( "=" * 70 )
			print()
			print( f"Fact:		   { fact }" )
			print( f"Stem/Question:  { stem }" )
			print( "Choices text:" )
			for i, c_text in enumerate( choices ):
				print( f"  [ { idx_to_tag[ i ] } ] { c_text }" )
			print( "-" * 70 )
			print( f"Generated out:  { gen_text }" )
			print( f"BERTScores:	 A: { scores_tensor[ 0 ]:.4f} | B: { scores_tensor[ 1 ]:.4f} | C: { scores_tensor[ 2 ]:.4f} | D: { scores_tensor[ 3 ]:.4f}" )
			print( f"Predicted tag:  { pred_tag }" )
			print( f"Reference tag:  { gt_tag } -> ( { 'CORRECT' if pred_tag == gt_tag else 'INCORRECT' } )" )
			print( "=" * 70 + "\n" )

	eval_accuracy = correct_mappings / total_samples
	print( f"eval complete -> mapping accuracy: { eval_accuracy * 100:.2f}%" )
	return eval_accuracy

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument( "-loadname", type=str, default="saved/model_best.pt" )
	parser.add_argument( "-valid_file", type=str, default="saved/mixture_valid.txt" )
	parser.add_argument( "-tokenizer_dir", type=str, default="saved/tokenizer" )

	parser.add_argument( "-d_model", type=int, default=1024 )
	parser.add_argument( "-d_ff", type=int, default=4096 )
	parser.add_argument( "-n_layers", type=int, default=16 )
	parser.add_argument( "-heads", type=int, default=16 )
	parser.add_argument( "-seqlen", type=int, default=1024 )
	parser.add_argument( "-batchsize", type=int, default=1 )

	parser.add_argument( "-dropout", type=float, default=0.0 )
	parser.add_argument( "-epsilon", type=float, default=1e-5 )
	parser.add_argument( "-epochs", type=int, default=1 )
	parser.add_argument( "-lr", type=float, default=2e-5 )

	parser.add_argument( "-train_path", type=str, default="obqa/obqa.train.txt" )
	parser.add_argument( "-valid_path", type=str, default="obqa/obqa.valid.txt" )
	parser.add_argument( "-test_path" , type=str, default="obqa/obqa.test.txt" )

	parser.add_argument( "-no_cuda", action="store_true" )

	parser.add_argument( 
		"-mode", 
		type=str, 
		choices=[ "cls", "gen" ], 
		default="",
		help="whether to run multiple-choice classification or autoregressive sequence generation."
	)
	parser.add_argument( 
		"-task_type", 
		type=str,
		choices=[ "ZS", "FT" ], 
		default="FT",
		help="zero-shot skips training and evaluates the base weights. fine-tuned trains the model first."
	)

	opt = parser.parse_args()
	
	device = torch.device( 
		"cuda:0" \
		if torch.cuda.is_available() and not opt.no_cuda \
		else "cpu"
	)

	tokenizer = GPT2TokenizerFast.from_pretrained( opt.tokenizer_dir )
	tokenizer.model_max_length = 10**9

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	state_dict = load_model_best_state_dict( opt.loadname )
	vocab_size = int( state_dict[ "wte.weight" ].shape[ 0 ] )

	model = TransformerGPT( 
		vocab_size,
		opt.d_model, opt.n_layers, 
		opt.heads, opt.seqlen, opt.d_ff, 
		opt.dropout, opt.epsilon
	)
	model.load_state_dict( state_dict, strict=True )
	model.to( device )

	if len( opt.mode ) == 0:
		print( "no mode indicated - validating base model" )
		model.eval()
		indices = my_tokenizer( opt.valid_file,tokenizer,1000000 )
		ppl = test_model( model=model, indices=indices, opt=opt, epoch=0, device=device )
		exit( 0 )

	obqa_train_raw = read_obqa( opt.train_path )
	obqa_valid_raw = read_obqa( opt.valid_path )
	obqa_test_raw  = read_obqa( opt.test_path )

	if opt.mode == "cls":
		print( "multiple-choice classification" )
		train_set = OBQADataset( obqa_train_raw, tokenizer )
		valid_set = OBQADataset( obqa_valid_raw, tokenizer )
		test_set  = OBQADataset( obqa_test_raw,  tokenizer )
		eval_batchsize = opt.batchsize 
	else:
		print( "autoregressive generation with beam search" )
		train_set = OBQAGenerationDataset( obqa_train_raw, tokenizer )
		valid_set = OBQAGenerationDataset( obqa_valid_raw, tokenizer )
		test_set  = OBQAGenerationDataset( obqa_test_raw,  tokenizer )
		# enforce batch size 1 for beam generation
		eval_batchsize = 1

	train_ldr = DataLoader( train_set, batch_size=opt.batchsize, shuffle=True )
	valid_ldr = DataLoader( valid_set, batch_size=opt.batchsize, shuffle=False )
	test_ldr  = DataLoader( test_set,  batch_size=opt.batchsize, shuffle=False )

	if opt.task_type == "ZS":
		print( f"--- zero-shot evaluation ---" )

		if opt.mode == "cls":
			valid_acc = eval_mcqa( model, valid_ldr, device )
		else:
			valid_acc = eval_generative( 
				model=model, 
				data_list=obqa_valid_raw, 
				tokenizer=tokenizer, 
				dev=device, 
				num_samples_to_print=3
			)
			
		print( f"zero-shot baseline valid. acc.: { valid_acc * 100:.2f}%" )

	elif opt.task_type == "FT":
		print( f"--- fine-tuning ---" )
		optimizer = torch.optim.AdamW( model.parameters(), lr=opt.lr )
		best_valid_acc = 0.0	

		print( "start training" )
		for epoch in range( opt.epochs ):
			if opt.mode == "cls":
				train_loss, train_acc = train_mcqa( model, train_ldr, optimizer, device )
				valid_acc = eval_mcqa( model, valid_ldr, device )
				print( f"epoch { epoch+1 } | train loss: { train_loss:.4f} | train acc: { train_acc*100:.2f}% | valid acc: { valid_acc*100:.2f}%" )
			else:
				train_loss, train_acc = train_generative( model, train_ldr, optimizer, device, beam_width=3, temperature=0.7 )
				valid_acc = eval_generative( 
					model=model, 
					data_list=obqa_valid_raw, 
					tokenizer=tokenizer, 
					dev=device, 
					num_samples_to_print=3
				)

				print( f"epoch { epoch+1 } | train loss: { train_loss:.4f} | train acc: { train_acc*100:.2f}% | valid BERTScore acc: { valid_acc*100:.2f}%" )

			if valid_acc > best_valid_acc:
				best_valid_acc = valid_acc
				save_path = f"aster_{ opt.mode }_obqa.pt"
				torch.save( { "model_state_dict": model.state_dict() }, save_path )
				print( f"new best model weights saved to { save_path }" )

		print( "start test" )

		if opt.mode == "cls":
			test_acc = eval_mcqa( model, test_ldr, device )
		else:
			test_acc = eval_generative( 
				model=model, 
				data_list=obqa_test_raw, 
				tokenizer=tokenizer, 
				dev=device, 
				num_samples_to_print=3
			)

		print( f"test acc: { test_acc:.4f}" )

if __name__ == "__main__":
	main()


