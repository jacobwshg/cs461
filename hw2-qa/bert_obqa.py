
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
import os
import argparse

try:
	import torch.amp as amp
except:
	import torch.cuda.amp as amp


SAVE_PATH = "./best_bert_obqa.pt"

class OBQADataset( Dataset ):
	def __init__( self, path, tokenizer, max_len=512 ):

		self.tokenizer = tokenizer
		self.max_len   = max_len
		self.data      = self.load_data( path )

	def load_data( self, path ):

		with open( path, "r" ) as f:
			entries = f.readlines()
	
		instances = []
		for ent in entries:
			fact, stem, ch_a, ch_b, ch_c, ch_d, ans_tag = tuple( ent.split( "|" ) )

			choice_texts = [ ch_a, ch_b, ch_c, ch_d ]

			ans_id = ord( ans_tag[ 0 ] ) - ord( "A" )

			instances.append(
				{
					"fact": fact,
					"stem": stem,
					"choices": choice_texts,
					"answer_id": ans_id
				}
			)

		return instances

	def __len__( self ):
		return len( self.data )

	def __getitem__( self, idx ):
		item = self.data[ idx ]

		fact      = item[ "fact" ]
		stem      = item[ "stem" ]
		choices   = item[ "choices" ]
		answer_id = item[ "answer_id" ]

		##"""
		# prepare "[CLS] <fact> <stem> <choice_text> [SEP]"
		inputseqs = [ f"{ fact } { stem } { ch }" for ch in choices ]

		def encode_seq( seq ):
			# let tokenizer add [CLS] and [SEP]
			return self.tokenizer(
				seq,
				add_special_tokens=True,
				max_length=self.max_len,
				padding="max_length",
				truncation=True,
				return_tensors="pt"
			)

		enc_inputs = [ encode_seq( seq ) for seq in inputseqs ]
		##"""

		"""
		enc_inputs = []
		for c in choices:
			enc_seq = self.tokenizer(
				text="{ fact } { stem }", text_pair=c,
				add_special_tokens=True,
				max_length=self.max_len,
				padding="max_length",
				truncation=True,
				return_tensors="pt"
			)
			enc_inputs.append( enc_seq )
		"""

		input_ids = torch.stack( [ enc[ "input_ids" ].squeeze( 0 ) for enc in enc_inputs ] )
		attn_msks = torch.stack( [ enc[ "attention_mask" ].squeeze( 0 ) for enc in enc_inputs ] )

		return \
		{
			"input_ids": input_ids,
			"attn_msk" : attn_msks,
			"labels": torch.tensor( answer_id, dtype=torch.long )
		}

class BertOBQA( nn.Module ):
	def __init__( self, bert_model_name="bert-base-uncased", dropout_rate=0.1 ):
		super( BertOBQA, self ).__init__()

		self.bertmodel = BertModel.from_pretrained( bert_model_name )

		self.dropout = nn.Dropout( dropout_rate )
		self.hidden_sz = self.bertmodel.config.hidden_size

		# output layer to score choices
		self.score_layer = nn.Linear( self.hidden_sz, 1 )

	# forward a batch of samples through BERT
	def forward( self, input_ids, attn_msk ):
		"""
		input_ids: ( batch_sz, num_choices, seq_len )
		attn_msk: ( batch_sz, num_choices, seq_len )

		returns:
		choice_scores ( logits ): ( batch_sz, num_choices )
		"""
		batch_sz, num_choices, seq_len = input_ids.shape

		input_ids_flat = input_ids.view( -1, seq_len )
		attn_msk_flat = attn_msk.view( -1, seq_len )

		outputs = self.bertmodel(
			input_ids=input_ids_flat,
			attention_mask=attn_msk_flat
		)

		# get [CLS] token embeddings ( idx 0 in seq embeddings -> first token )
		cls_embeds = outputs.last_hidden_state[ :, 0, : ]  # ( batch_sz * num_choices, hidden_sz )
		cls_embeds = self.dropout( cls_embeds )

		choice_scores = self.score_layer( cls_embeds )  # ( batch_sz * num_choices, 1 )
		# reshape back to ( batch_sz, num_choices )
		choice_scores = choice_scores.view( batch_sz, num_choices )

		return choice_scores

def train_model(
	model,
	train_loader, valid_loader,
	save_path=SAVE_PATH,
	num_epochs=2, lr=2e-5, 
	dev="cuda"
):

	model.to( dev )

	optimizer = AdamW( model.parameters(), lr=lr, weight_decay=0.01 )
	loss_fn = nn.CrossEntropyLoss()

	best_valid_acc = 0.0

	for epoch in range( num_epochs ):
		print( f"\nepoch {epoch + 1}/{num_epochs}" )
		
		# training phase
		model.train()
		total_train_loss = 0
		train_correct = 0
		train_total = 0

		scaler = amp.GradScaler()

		progbar = tqdm( train_loader, desc=f"training epoch {epoch+1}" )
		for batch in progbar:

			input_ids = batch[ "input_ids" ].to( dev )
			attn_msk  = batch[ "attn_msk" ].to( dev )
			labels    = batch[ "labels" ].to( dev )

			optimizer.zero_grad()

			with torch.autocast(
				device_type="cuda" if "cuda" in str( dev ) else "cpu",
				dtype=torch.float16
			):
				# forward
				choice_scores = model( input_ids, attn_msk )
				loss = loss_fn( choice_scores, labels )

			# backward
			scaler.scale( loss ).backward()
			scaler.step( optimizer )
			scaler.update()

			# compute accuracy
			preds = torch.argmax( choice_scores.float(), dim=1 )
			train_correct += ( preds == labels ).sum().item()
			train_total += labels.size( 0 )
			total_train_loss += loss.item()

			progbar.set_postfix( 
				{
					"loss": loss.item(),
					"acc" : train_correct / train_total
				}
			)

		avg_train_loss = total_train_loss / len( train_loader )
		train_acc = train_correct / train_total

		# validation
		valid_acc = eval_model( model, valid_loader, dev )

		print( f"train loss: { avg_train_loss:.4f}, train acc: { train_acc:.4f}, valid acc: { valid_acc:.4f}"  )
		
		# save best model
		if valid_acc > best_valid_acc:
			best_valid_acc = valid_acc
			torch.save( model.state_dict(), save_path )
			print( f"new best model saved with valid acc: { valid_acc:.4f}" )

@torch.no_grad()
def eval_model( model, data_loader, dev="cuda" ):
	"""
	evaluate model on validation/test set
	"""
	model.eval()
	correct = 0
	total = 0

	with torch.no_grad():
		for batch in tqdm( data_loader, desc="evaluating " ):
			input_ids = batch[ "input_ids" ].to( dev )
			attn_msk = batch[ "attn_msk" ].to( dev )
			labels = batch[ "labels" ].to( dev )

			choice_scores = model( input_ids, attn_msk )
			preds = torch.argmax( choice_scores, dim=1 )

			correct += ( preds == labels ).sum().item()
			total += labels.size( 0 )

	acc = correct / total if total > 0 else 0
	return acc

def predict( model, data_loader, dev="cuda" ):
	"""
	predict on test set
	"""
	model.eval()
	preds = []

	with torch.no_grad():
		for batch in tqdm( data_loader, desc="predicting" ):
			input_ids = batch[ "input_ids" ].to( dev )
			attn_msk = batch[ "attn_msk" ].to( dev )

			choice_scores = model( input_ids, attn_msk )
			batch_preds = torch.argmax( choice_scores, dim=1 )

			preds.extend( batch_preds.cpu().numpy() )

	return preds

def main():

	BASE_MODEL_NAME = "bert-base-uncased"

	parser = argparse.ArgumentParser()

	parser.add_argument( "-save_path", type=str, default=SAVE_PATH )
	parser.add_argument( "-max_len",   type=int, default=256 )
	parser.add_argument( "-batchsize", type=int, default=4 )
	parser.add_argument( "-epochs",    type=int, default=1 )
	parser.add_argument( "-lr",        type=float, default=2e-5 )
	parser.add_argument( "-task_type", type=str, default="ZS", help="ZS for zero-shot, FT for fine-tuned" )
	parser.add_argument( "-train_path", type=str, default="obqa/obqa.train.txt" )
	parser.add_argument( "-valid_path", type=str, default="obqa/obqa.valid.txt" )
	parser.add_argument( "-test_path",  type=str, default="obqa/obqa.test.txt" )

	opt = parser.parse_args()

	dev = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )

	print( f"using device: { dev }" )
	print( f"task type:", opt.task_type )
	print( f"save path ( if fine-tuning ): ", opt.save_path )
	print( f"training epochs ( if fine-tuning ): ", opt.epochs )
	print( f"batchsize: ", opt.batchsize )

	# initialize
	model = BertOBQA( BASE_MODEL_NAME )
	model.to( dev )
	tokenizer = BertTokenizer.from_pretrained( BASE_MODEL_NAME )

	# add special tokens if needed ( though we're using existing ones )
	tokenizer.add_special_tokens( { "pad_token": "[PAD]" } )

	# instantiate datasets
	train_set = OBQADataset(
		path=opt.train_path,
		tokenizer=tokenizer, 
		max_len=opt.max_len
	)
	valid_set = OBQADataset(
		path=opt.valid_path,
		tokenizer=tokenizer, 
		max_len=opt.max_len
	)
	test_set = OBQADataset(
		path=opt.test_path,
		tokenizer=tokenizer, 
		max_len=opt.max_len
	)

	# create data loaders
	train_loader = DataLoader( train_set, batch_size=opt.batchsize, shuffle=True )
	valid_loader = DataLoader( valid_set, batch_size=opt.batchsize, shuffle=False )
	test_loader  = DataLoader( test_set, batch_size=opt.batchsize, shuffle=False )

	if opt.task_type == "FT":

		# train 
		print( "start fine-tuned training" )
		train_model(
			model=model,
			train_loader=train_loader, valid_loader=valid_loader,
			save_path=opt.save_path,
			num_epochs=opt.epochs, lr=opt.lr,
			dev=dev
		)

		# load best model for final evaluation
		model.load_state_dict( torch.load( opt.save_path ) )

	else:
		print( "zero-shot evaluation" )

	# eval on validation set
	valid_acc = eval_model( model, valid_loader, dev )
	print( f"final validation accuracy: { valid_acc:.4f}" )

	# eval on test set
	test_acc = eval_model( model, test_loader, dev )
	print( f"final test accuracy: { test_acc:.4f}" )

	"""
	# predict on test set
	test_preds = predict( model, test_loader, dev )
	#print( f"test predictions shape: { len( test_predictions ) }" )

	# remap answer_ids back to letters
	pred_tags = [ chr( ord( "A" ) + pred ) for pred in test_preds ]

	# save results
	with open( "predictions.json", "w" ) as f:
		json.dump( pred_tags, f )
	"""

	print( "train and eval complete" )

if __name__ == "__main__":
	main()

