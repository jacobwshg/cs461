
import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

class GPT2Attention(nn.Module):
	def __init__(self, d_model, heads, max_seq_len, attn_dropout=0.0, resid_dropout=0.0):
		super().__init__()

		assert d_model % heads == 0

		self.heads = heads
		self.d_model = d_model
		self.head_dim = d_model // heads

		self.c_attn = nn.Linear(d_model, 3 * d_model)
		self.c_proj = nn.Linear(d_model, d_model)

		self.attn_dropout = nn.Dropout(attn_dropout)
		self.resid_dropout = nn.Dropout(resid_dropout)

		bias = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)).view(1, 1, max_seq_len, max_seq_len)
		self.register_buffer("bias", bias, persistent=False)

	def forward(self, x, attention_mask=None):
		bsz, seq_len, _ = x.size()

		qkv = self.c_attn(x)
		q, k, v = qkv.split(self.d_model, dim=2)

		q = q.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
		k = k.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
		v = v.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

		scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

		causal_mask = self.bias[:, :, :seq_len, :seq_len]
		scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

		if attention_mask is not None:
			key_mask = attention_mask[:, None, None, :].to(torch.bool)
			scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

		probs = F.softmax(scores, dim=-1)
		probs = self.attn_dropout(probs)

		attn = torch.matmul(probs, v)
		attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)

		return self.resid_dropout(self.c_proj(attn))


class GPT2MLP(nn.Module):
	def __init__(self, d_model, d_ff, resid_dropout=0.0):
		super().__init__()

		self.c_fc = nn.Linear(d_model, d_ff)
		self.c_proj = nn.Linear(d_ff, d_model)

		try:
			self.act = nn.GELU(approximate="tanh")
		except TypeError:
			self.act = nn.GELU()

		self.dropout = nn.Dropout(resid_dropout)

	def forward(self, x):
		return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class GPT2Block(nn.Module):
	def __init__(self, d_model, heads, d_ff, max_seq_len, attn_dropout=0.0, resid_dropout=0.0, layer_norm_epsilon=1e-5):
		super().__init__()

		self.ln_1 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
		self.attn = GPT2Attention(d_model, heads, max_seq_len, attn_dropout=attn_dropout, resid_dropout=resid_dropout)

		self.ln_2 = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
		self.mlp = GPT2MLP(d_model, d_ff, resid_dropout=resid_dropout)

	def forward(self, x, attention_mask=None):
		x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
		x = x + self.mlp(self.ln_2(x))
		return x

class TransformerGPT(nn.Module):
	def __init__(self, vocab_size, d_model, n_layers, heads, seqlen, d_ff, dropout=0.0, layer_norm_epsilon=1e-5):
		super().__init__()

		self.seqlen = seqlen

		self.wte = nn.Embedding(vocab_size, d_model)
		self.wpe = nn.Embedding(seqlen, d_model)

		self.drop = nn.Dropout(dropout)

		self.h = nn.ModuleList([GPT2Block(d_model, heads, d_ff, seqlen, attn_dropout=dropout, resid_dropout=dropout, layer_norm_epsilon=layer_norm_epsilon) for _ in range(n_layers)])

		self.ln_f = nn.LayerNorm(d_model, eps=layer_norm_epsilon)

		self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
		self.lm_head.weight = self.wte.weight

		self.classification_head = nn.Sequential(
			nn.Linear(d_model, d_model),
			nn.Tanh(),
			nn.Dropout(dropout),
			nn.Linear(d_model, num_classes)
		)

	def forward(self, input_ids, attention_mask=None, task="lm"):
		bsz, seq_len = input_ids.size()

		if seq_len > self.seqlen:
			raise ValueError(f"sequence length {seq_len} exceeds model seqlen {self.seqlen}")

		pos = torch.arange(0, seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)

		x = self.wte(input_ids) + self.wpe(pos)
		x = self.drop(x)

		for block in self.h:
			x = block(x, attention_mask=attention_mask)

		x = self.ln_f(x)

		if task == "lm":
			logits = self.lm_head(x)
			return x, logits
		elif task == "cls":
			# Pool the last token's representation (standard for decoder-only MCQA)
			pooled = x[:, -1, :]  # shape: [batch, d_model]
			cls_logits = self.classification_head(pooled)  # shape: [batch, num_classes]
			return x, cls_logits


def load_model_best_state_dict(path):
	checkpoint = torch.load(path, map_location="cpu")

	if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
		state_dict = checkpoint["model_state_dict"]
	elif isinstance(checkpoint, dict) and "model" in checkpoint:
		state_dict = checkpoint["model"]
	else:
		state_dict = checkpoint

	return state_dict

def my_tokenizer(path, tokenizer, max_tokens):
	indices = []
	line_batch = []

	with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
		for line in f:
			encoded = tokenizer(line, add_special_tokens=False)["input_ids"]
			if len(indices) < max_tokens:
				for ids in encoded:
					indices.append(int(ids))
	return indices

@torch.no_grad()
def test_model(model, indices, opt, epoch=0, device=None):
	start_time = time.time()

	if device is None:
		device = next(model.parameters()).device

	aa = opt.eval_seqlen
	bb = opt.eval_batchsize
	total_loss = 0.0
	count = 0

	n_tokens = len(indices)
	vocab_size = model.wte.weight.size(0)
	stride = aa * bb

	with torch.no_grad():
		for i in range(0, n_tokens - aa + 1, stride):
			src = torch.zeros((bb, aa), dtype=torch.long)
			trg = torch.zeros((bb, aa - 1, vocab_size), dtype=torch.float)
			actual_batchsize = 0
			for k in range(bb):
				start_idx = i + k * aa
				if start_idx + aa > n_tokens:
					break
				for j in range(aa-1):
					src[k, j] = indices[start_idx + j]
					next_token_id = indices[start_idx + j + 1]
					trg[k, j, indices[start_idx + j + 1]] = 1.0

				actual_batchsize += 1

			src = src[:actual_batchsize].to(device)
			trg = trg[:actual_batchsize].to(device)

			attention_mask = torch.ones_like(src, dtype=torch.long, device=device)

			x,preds = model(src, attention_mask=attention_mask)

			preds = preds[:, :-1, :]

			max_preds = torch.amax(preds, dim=2).unsqueeze(2)
			preds = preds - max_preds
			logits = torch.exp(preds)
			denoms = torch.sum(logits, 2)
			denoms = denoms.unsqueeze(2)
			numer = logits * trg
			numer = torch.sum(numer, 2)
			numer = numer.unsqueeze(2)
			probs = numer / denoms
			loss = -torch.log(probs + 1e-12).mean()
			print(i,loss.item())

			total_loss += loss.item()
			count += 1

	avg_loss = total_loss / count
	ppl = math.exp(min(avg_loss, 20.0))

	elapsed_min = int((time.time() - start_time) // 60)

	print(" ")
	print("%dm: TEST %d [%s]  100%%  loss = %.3f" % (elapsed_min, epoch + 1, "#" * 20, avg_loss))
	print("epoch %d complete, loss = %.03f ppl = %7.1f" % (epoch + 1, avg_loss, ppl))
	print(" ")

	return ppl

def read_obqa(file_name):
	data = []
	with open(file_name,'rt') as f:
		for line in f:
			line = line.replace('\n','')
			tokens = line.split('|')
			d = {}
			d['fact'] = tokens[0]
			d['stem'] = tokens[1]
			d['A'] = tokens[2]
			d['B'] = tokens[3]
			d['C'] = tokens[4]
			d['D'] = tokens[5]
			d['Answer'] = tokens[6]   
			data.append(d)
	for i in range(5):
		print(i,data[i])
	print('data: %d' % (len(data)))
	return(data)

def format_qa_instance(sample):
	prompt = (
		f"Fact: {sample['fact']} | "
		f"Question: {sample['stem']} | "
		f"A: {sample['A']} | B: {sample['B']} | "
		f"C: {sample['C']} | D: {sample['D']} | Answer:"
	)
	label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
	label = label_map[sample['Answer'].strip()]
	return prompt, label


class MCQADataset(Dataset):
	def __init__(self, samples, tokenizer, max_len):
		self.samples = samples
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __len__(self): return len(self.samples)

	def __getitem__(self, idx):
		text, label = format_qa_instance(self.samples[idx])
		enc = self.tokenizer(
			text, max_length=self.max_len, truncation=True, 
			padding="max_length", return_tensors="pt"
		)
		return {
			"input_ids": enc["input_ids"].squeeze(0),
			"attention_mask": enc["attention_mask"].squeeze(0),
			"labels": torch.tensor(label, dtype=torch.long)
		}

def train_qa(model, dataloader, optimizer, scheduler, device, epochs=3):
	model.train()
	criterion = nn.CrossEntropyLoss()

	for epoch in range(epochs):
		total_loss, correct, total = 0, 0, 0
		progbar = tqdm( dataloader, desc="train QA" )
		for batch in progbar:
			input_ids = batch["input_ids"].to(device)
			attn_mask = batch["attention_mask"].to(device)
			labels = batch["labels"].to(device)
			
			optimizer.zero_grad()
			_, cls_logits = model(input_ids, attention_mask=attn_mask, task="cls")
			
			loss = criterion(cls_logits, labels)
			loss.backward()
			optimizer.step()
			scheduler.step()
			
			total_loss += loss.item()
			preds = cls_logits.argmax(dim=-1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
			
		avg_loss = total_loss / len(dataloader)
		acc = correct / total
		print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

@torch.no_grad()
def eval_qa(model, dataloader, device):
	model.eval()
	correct, total = 0, 0

	progbar = tqdm( dataloader, desc="eval QA" )
	for batch in dataloader:
		input_ids = batch["input_ids"].to(device)
		attn_mask = batch["attention_mask"].to(device)
		labels = batch["labels"].to(device)
		
		_, cls_logits = model(input_ids, attention_mask=attn_mask, task="cls")
		preds = cls_logits.argmax(dim=-1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)
	return correct / total

def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("-loadname", type=str, default="")
	parser.add_argument("-valid_file", type=str, default="")
	parser.add_argument("-tokenizer_dir", type=str, default="")
	parser.add_argument("-d_model", type=int, default=1024)
	parser.add_argument("-d_ff", type=int, default=4096)
	parser.add_argument("-n_layers", type=int, default=16)
	parser.add_argument("-heads", type=int, default=16)
	parser.add_argument("-seqlen", type=int, default=1024)
	parser.add_argument("-eval_seqlen", type=int, default=None)
	parser.add_argument("-eval_batchsize", type=int, default=1)
	parser.add_argument("-dropout", type=float, default=0.0)
	parser.add_argument("-epsilon", type=float, default=1e-5)
	parser.add_argument("-no_cuda", action="store_true")

	parser.add_argument("-train_file", type=str, default="obqa.train.txt")
	parser.add_argument("-valid_file", type=str, default="obqa.valid.txt")
	parser.add_argument("-lr", type=float, default=5e-5)

	opt = parser.parse_args()

	if opt.eval_seqlen is None:
		opt.eval_seqlen = opt.seqlen

	device = torch.device("cuda:0" if torch.cuda.is_available() and not opt.no_cuda else "cpu")

	obqa_train = read_obqa(opt.train_file)
	obqa_valid = read_obqa(opt.valid_file)

	tokenizer = GPT2TokenizerFast.from_pretrained(opt.tokenizer_dir)
	tokenizer.model_max_length = 10**9

	state_dict = load_model_best_state_dict(opt.loadname)
	vocab_size = int(state_dict["wte.weight"].shape[0])

	model = TransformerGPT(vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.seqlen, opt.d_ff, 
						   opt.dropout, opt.epsilon)
	try:
		model.load_state_dict(state_dict, strict=True)
	except:
		model_dict = model.state_dict()
		pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		model.load_state_dict(model_dict)

	model.to(device)

	train_dataset = MCQADataset(obqa_train, tokenizer, opt.seqlen)
	valid_dataset = MCQADataset(obqa_valid, tokenizer, opt.seqlen)
	
	train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
	valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

	optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
	

	train_qa(model, train_loader, optimizer, scheduler, device, epochs=3)
	
	acc = eval_qa( model, dataloader, device )
	print( f"eval acc: { acc:.2f}" )	
	#model.eval()

	#indices = my_tokenizer(opt.valid_file,tokenizer,1000000)
	#ppl = test_model(model=model, indices=indices, opt=opt, epoch=0, device=device)

if __name__ == "__main__":
	main()

