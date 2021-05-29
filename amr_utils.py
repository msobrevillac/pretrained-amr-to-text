from constants import SPECIAL_TOKENS, MODEL_INPUTS, AMR_SPECIAL_TOKENS
from itertools import chain
from collections import defaultdict
import torch
from amr import AMR
from torch.utils.data import DataLoader, TensorDataset
from utils import update_model


def load_amr(src_path, tgt_path=None, generation=True):
	amr = AMR(generation) 
	with open(src_path, "r") as f:
		print(src_path)
		amr.source = [line.strip() for line in f]
  
	if tgt_path is not None:
		with open(tgt_path, "r") as f:
			amr.target = [line.strip() for line in f]
	amr.extract_edges()
	return amr


def tokenize_amr(tokenizer, dataset):
	encoded_dataset = []
	'''
		Example
		#text0 = "haver-02 :polarity - :arg1 entregar-01 :location país :name nome :op1 \"brasil\""
		#text0 = "(h / haver-02 :polarity - :arg1 (e / entregar-01) :location (c / país :name (n / nome :op1 \"brasil\")))"
		print(text0, "\n", tokenizer.tokenize(text0), "\n", tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text0)))
 	'''
	for data_inst in dataset:  
		tok_amr = tokenizer.convert_tokens_to_ids(
										tokenizer.tokenize(data_inst[0]))
		if data_inst[1] is not None:
			tok_txt = tokenizer.convert_tokens_to_ids(
										tokenizer.tokenize(data_inst[1]))
		else:
			tok_txt = None
		encoded_dataset.append((tok_amr, tok_txt))
	return encoded_dataset



def pre_process_amr_leftpad( amr_graph, text, tokenizer, max_input_length, with_text=True, with_masking=True):
	bos, eos, ctx, ans, que, pad, gen = \
		tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

	padded = []

	amr = [bos] + amr_graph + [gen]
	if text is not None:
		text = (text + [eos] if with_text else [])
	else:
		text = []

	max_len = int((max_input_length-3)/2)

	if len(amr) > max_len:
		amr = amr[:max_len]
		amr[-1] = gen
	if len(text) > max_len:
		text = text[:max_len]

	combined = list(chain(amr, text))
	len_combined = len(combined)

	if len_combined < max_input_length:
		len_reamining = max_input_length - len_combined
		padded = [pad] * len_reamining

	instance = {}
	instance["input_ids"] = list(chain(padded, amr, text))
	instance["token_type_ids"] = [pad] * len(padded) + [ctx]   \
		* len(amr) + [ans] * len(text)
	instance["attention_mask"] = [0]*len(padded) \
		+ [1]*(max_input_length-len(padded))

	if with_masking:
		instance["labels"] = [-100] * (len(padded) + len(amr)) + text
	else:
		instance["labels"] = [-100] * len(padded) + list(chain(amr, text))

	return instance


def preproc_amr (tokenizer, encoded_dataset, with_text=True, max_length=80, with_masking=False):
	datasets = defaultdict(list)
	for idx, (amr_graph, text) in enumerate(encoded_dataset):
		instance_que = pre_process_amr_leftpad(
			amr_graph,
			text,
			tokenizer,
			max_length,#args.max_input_length,
			with_text=with_text,
			with_masking=with_masking)#with_masking=args.with_masking,
		for input_name, input_array in instance_que.items():
			datasets[input_name].append(input_array)

	tensor_datasets = []
	datasets_padded = datasets
	for input_name in MODEL_INPUTS:
		padded = datasets_padded[input_name]
		tensor_datasets.append(torch.tensor(padded))

	return tensor_datasets


def get_data_loaders(src_path, tokenizer, model, tgt_path=None, shuffle=True, batch_size=16, is_train=True, max_length=80, generation=True):
	if tgt_path is None:
		amr = load_amr(src_path, generation=generation)
	else:
		amr = load_amr(src_path, tgt_path, generation)
  
	if is_train:
		update_model(tokenizer, model, amr)

	dataset = []
	if tgt_path is not None:
		for src, tgt in zip(amr.source, amr.target):    
			dataset.append((src,tgt))
	else:
		for src in amr.source:    
			dataset.append((src,None))
  
	encoded_dataset = tokenize_amr(tokenizer, dataset)

	tensor_datasets = \
			preproc_amr(tokenizer, encoded_dataset, max_length=max_length, with_text=True) \
			+ preproc_amr(tokenizer, encoded_dataset, max_length=max_length, with_text=False)
	dataset = TensorDataset(*tensor_datasets)

	loader = DataLoader(dataset, batch_size=batch_size,
			shuffle=shuffle, pin_memory=False)
	return loader
