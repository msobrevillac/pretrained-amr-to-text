import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from constants import UD_SPECIAL_PREFIX, CG_SPECIAL_PREFIX, GRAPH_SPECIAL_PREFIX

class WebnlgDataset(Dataset):

	def __init__(self, dataframe, tokenizer, source_len, target_len):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.source_len = source_len
		self.target_len = target_len
		self.source = self.data.source
		if 'target' in dataframe.columns:
			self.target = self.data.target
		else:
			self.target = None

	def __len__(self):
		return len(self.source)

	def __getitem__(self, index):
		str_source = str(self.source[index]).strip()
		source = self.tokenizer.batch_encode_plus([str_source], max_length= self.source_len, 
			pad_to_max_length=True, return_tensors='pt', truncation=True)
		source_ids = source['input_ids'].squeeze()
		source_mask = source['attention_mask'].squeeze()

		if self.target is not None:
			str_target = str(self.target[index]).strip()
			target = self.tokenizer.batch_encode_plus([str_target], max_length= self.target_len,
				pad_to_max_length=True, return_tensors='pt', truncation=True)
			target_ids = target['input_ids'].squeeze()
			target_mask = target['attention_mask'].squeeze()
			return {
				'source_ids': source_ids.to(dtype=torch.long), 
				'source_mask': source_mask.to(dtype=torch.long), 
				'target_ids': target_ids.to(dtype=torch.long),
				'target_ids_y': target_ids.to(dtype=torch.long)
				}
		else:
			return {
				'source_ids': source_ids.to(dtype=torch.long), 
				'source_mask': source_mask.to(dtype=torch.long)
				}


class AMRDataset(Dataset):

	def __init__(self, dataframe, tokenizer, src_max_len, tgt_max_len, \
				pretrained="gpt2", train=False, generation=True):
		self.tokenizer = tokenizer
		self.data = dataframe
		self.source_len = src_max_len
		self.target_len = tgt_max_len
		self.source = self.data.source
		if 'target' in dataframe.columns:
			self.target = self.data.target
		else:
			self.target = None
		self.pretrained = pretrained
		if train:
			self.add_special_tokens(generation)

	def __len__(self):
		return len(self.source)

	def add_special_tokens(self, generation):
		if generation:
			source = self.source
		else:
			source = self.target

		special_tokens = [tok for src in source for tok in src.split() if tok.startswith(":") \
								and tok not in self.tokenizer.get_added_vocab()]

		if self.pretrained.endswith("t5"):

			for gsp in GRAPH_SPECIAL_PREFIX:
				if gsp not in self.tokenizer.get_added_vocab():
					special_tokens.append(gsp)

		if self.pretrained.endswith("bart"):
			special_tokens += ["amr_AMR","pt_XX"]

		if len(special_tokens) > 0:
			self.tokenizer.add_tokens(list(set(special_tokens)))

	def __getitem__(self, index):

		str_source = str(self.source[index]).strip()
		if self.pretrained.endswith("t5"):
			source = self.tokenizer.batch_encode_plus([str_source], max_length= self.source_len, 
				pad_to_max_length=True, return_tensors='pt', truncation=True)

		if self.pretrained.endswith("bart"):
			str_source = str_source + " </s> amr_AMR"
			source = self.tokenizer.batch_encode_plus([str_source], max_length= self.source_len, 
				pad_to_max_length=True, return_tensors='pt', truncation=True, add_special_tokens=False)


		source_ids = source['input_ids'].squeeze()
		source_mask = source['attention_mask'].squeeze()

		if self.target is not None:
			if self.pretrained.endswith("t5"):
				str_target = "<pad> " + str(self.target[index]).strip()
				target = self.tokenizer.batch_encode_plus([str_target], max_length= self.target_len,
					pad_to_max_length=True, return_tensors='pt', truncation=True)

			if self.pretrained.endswith("bart"):
				str_target = "pt_XX "  + str(self.target[index]).strip() + " </s>"
				target = self.tokenizer.batch_encode_plus([str_target], max_length= self.target_len,
					pad_to_max_length=True, return_tensors='pt', truncation=True, add_special_tokens=False)

			target_ids = target['input_ids'].squeeze()
			target_mask = target['attention_mask'].squeeze()
			return {
				'source_ids': source_ids.to(dtype=torch.long), 
				'source_mask': source_mask.to(dtype=torch.long), 
				'target_ids': target_ids.to(dtype=torch.long)#,
				#'target_ids_y': target_ids.to(dtype=torch.long)
				}
		else:
			return {
				'source_ids': source_ids.to(dtype=torch.long), 
				'source_mask': source_mask.to(dtype=torch.long)
				}



def process_data(src_path, tgt_path=None):
	src = []
	with open(src_path, "r") as f:
		src = [line.strip() for line in f]
  
	tgt = []
	if tgt_path is not None:
		with open(tgt_path, "r") as f:
			tgt = [line.strip() for line in f]
			return pd.DataFrame({'source': src, 'target': tgt})
	else:
		return pd.DataFrame({'source': src})

