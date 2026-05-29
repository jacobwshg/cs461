
#!/bin/bash

python3 Aster.py \
	-loadname saved/model_best.pt \
	-d_model 1024 \
	-d_ff 4096 \
	-n_layers 16 \
	-heads 16 \
	-seqlen 1024 \
	-batchsize 8 \
	-valid_file saved/mixture_valid.txt \
	-tokenizer saved/tokenizer 

