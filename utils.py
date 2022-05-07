import torch
import random
import numpy as np

def set_seed(seed):
	'''
		Setting a seed to make our experiments reproducible
		seed: seed value
	'''
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	torch.backends.cudnn.deterministic = True
	#torch.backends.cudnn.benchmark = False
	random.seed(seed)


def improved(metric, best_metric, criteria):
	if criteria == "perplexity":
		if metric < best_metric:
			return True
	else:
		if metric > best_metric:
			return True
	return False


def parse_corpus(fname):
	'''
		This method parses the AMR corpus extracting ids, sentences, alignments and AMR-annotations
	'''

	with open(fname) as f:
		doc = f.read()

	instances = doc.split('\n\n')[1:]
	amrs = []
	for instance in instances:
		try:
			instance = instance.split('\n')
			id_sentence = instance[0].strip()
			sentence_pt = (' '.join(instance[1].split()[2:])).strip()
			sentence_en = (' '.join(instance[2].split()[2:])).strip()
			alignments_bren = (' '.join(instance[3].split()[2:])).strip()
			alignments_bramr = (' '.join(instance[4].split()[2:])).strip()
			amr = ('\n'.join(instance[5:])).strip()

			amrs.append({'id':id_sentence, 'sentence_pt': sentence_pt, 'sentence_en': sentence_en, 'alignments_bren': alignments_bren, 'alignments_bramr': alignments_bramr, 'amr': amr})
		except:
			pass
	return amrs


def read_just_amr(fname):
	'''
		This method parses the AMR corpus extracting ids, sentences, alignments and AMR-annotations
	'''

	with open(fname) as f:
		doc = f.read()

	instances = doc.split('\n\n')
	amrs = []
	for instance in instances:
		try:
			amrs.append(instance)
		except:
			pass
	return amrs


def trim_pad(input_ids, lm_labels, token_type_ids, attention_mask, pad):
	min_idx = (input_ids != pad).nonzero()[:, 1].min()

	return [input_ids[:, min_idx:], lm_labels[:, min_idx:],
			token_type_ids[:, min_idx:], attention_mask[:, min_idx:]]


def trim_batch(batch, pad):
	input_ids, lm_labels, token_type_ids, attention_mask, partial_input_ids,\
				partial_lm_labels, partial_token_type_ids, partial_attention_mask = batch

	return trim_pad(input_ids, lm_labels, token_type_ids, attention_mask, pad)\
		+ trim_pad(partial_input_ids, partial_lm_labels,
				partial_token_type_ids, partial_attention_mask, pad)


def update_model(tokenizer, model, amr):
	tokenizer.add_tokens(amr.edges)
	model.resize_token_embeddings(len(tokenizer))


def apply_loss(idx, optimizer, loss,retain_graph=False):#(idx, optimizer, loss, args, retain_graph=False):
	#loss /= args.gradient_accumulation_steps
	loss.backward(retain_graph=retain_graph)
	#if args.max_norm is not None:
	#  params = optimizer.param_groups[0]['params']
	#  torch.nn.utils.clip_grad_norm_(params, args.max_norm)
	#if idx % args.gradient_accumulation_steps == 0:
	optimizer.step()
	optimizer.zero_grad()
	return loss


