from Dataloader import AMRDataset, process_data
from utils import set_seed, improved
import math
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import MBartForConditionalGeneration, MBartTokenizer
from constants import AMR_SPECIAL_TOKENS
import sacrebleu
import os

def predict(model, loader, tokenizer, max_length, beam_size, device):

	model.eval()
	predictions = []
	with torch.no_grad():
		for _, data in enumerate(loader, 0):
			ids = data['source_ids'].to(device, dtype = torch.long)
			mask = data['source_mask'].to(device, dtype = torch.long)

			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask,
				pad_token_id=tokenizer.pad_token_id,
				max_length=max_length, 
				num_beams=beam_size,
				no_repeat_ngram_size=2,
				num_return_sequences=1,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True,
				eos_token_id=tokenizer.eos_token_id)

			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]

			predictions.extend(preds)
	return predictions


def evaluate_bleu(model, loader, tokenizer, max_length, beam_size, device):

	model.eval()
	references = []
	hypothesis = []
	with torch.no_grad():
		for _, data in enumerate(loader, 0):
			ids = data['source_ids'].to(device, dtype = torch.long)
			mask = data['source_mask'].to(device, dtype = torch.long)
			y = data['target_ids'].to(device, dtype = torch.long)

			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask,
				pad_token_id=tokenizer.pad_token_id,
				max_length=max_length, 
				num_beams=beam_size,
				no_repeat_ngram_size=2,
				num_return_sequences=1,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True,
				eos_token_id=tokenizer.eos_token_id)

			refs = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in y]
			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in generated_ids]

			references.extend(refs)
			hypothesis.extend(preds)

	bleu = sacrebleu.corpus_bleu(hypothesis, [references])

	return bleu.score


def evaluate_loss(model, loader, tokenizer, device):

	model.eval()
	total_loss = 0
	n = 0
	for index, data in enumerate(loader, 0):
		ids = data['source_ids'].to(device, dtype = torch.long)
		mask = data['source_mask'].to(device, dtype = torch.long)

		y = data['target_ids'].to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

		outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels)
		loss = outputs[0]
		total_loss += loss.item()
		n += 1

	return total_loss / n


def train_epoch(model, epoch, loader, tokenizer, optimizer, print_every, device, accumulation_steps):
	total_loss = 0.0
	n = 0
	model.train()
	for index, data in enumerate(loader, 0):
		ids = data['source_ids'].to(device, dtype = torch.long)
		mask = data['source_mask'].to(device, dtype = torch.long)

		y = data['target_ids'].to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100

		outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, labels=lm_labels) #use_cache=False for MT5
		loss = outputs[0]
		loss = loss / accumulation_steps
		total_loss += loss.item()
		n += 1
        
		if (index+1) % print_every == 0:
			print(f'Epoch: {epoch} | Step: {index+1} | Loss: {total_loss/n}')
			n = 0
			total_loss = 0

		loss.backward(retain_graph=False)
		if (index+1) % accumulation_steps == 0:
			optimizer.step()
			optimizer.zero_grad()

	optimizer.step()
	optimizer.zero_grad()

def main(args):

	set_seed(args.seed)

	if args.parsing:
		generation = False
	else:
		generation = True


	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	df_train = process_data(args.train_source, args.train_target)
	df_dev = process_data(args.dev_source)
	df_test = process_data(args.test_source)


	tokenizer = MBartTokenizer.from_pretrained(args.model)

	if args.representation == "amr":
		tokenizer.add_tokens(AMR_SPECIAL_TOKENS)

	train_set = AMRDataset(df_train, tokenizer, args.src_max_length, args.tgt_max_length, \
				pretrained=args.pretrained_model, train=True, generation=generation)

	dev_set = AMRDataset(df_dev, tokenizer, args.src_max_length, args.tgt_max_length, \
				pretrained=args.pretrained_model)
	test_set = AMRDataset(df_test, tokenizer, args.src_max_length, args.tgt_max_length, \
				pretrained=args.pretrained_model)

	train_params = {'batch_size': args.batch_size,
				'shuffle': True,
				'num_workers': 0
				}

	dev_params = {'batch_size': args.batch_size,
				'shuffle': False,
				'num_workers': 0
				}

	test_params = {'batch_size': args.batch_size,
				'shuffle': False,
				'num_workers': 0
				}


	train_loader = DataLoader(train_set, **train_params)
	dev_loader = DataLoader(dev_set, **dev_params)
	test_loader = DataLoader(test_set, **test_params)

	model = MBartForConditionalGeneration.from_pretrained(args.model)
	if torch.cuda.device_count() > 1:
		print("Let's use", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)

	model = model.to(device)

	model.resize_token_embeddings(len(tokenizer))

	if args.fixed_embeddings:
		fixed_param = "model.shared.weight"
		for name, param in model.named_parameters():
			if name == fixed_param:
				param.requires_grad = False
				print("Freezing ", fixed_param)

	#if args.optimizer == "adam":
	optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

	if args.early_stopping_criteria == "perplexity":
		best_metric = float('inf')
	else:
		best_metric = 0.0

	patience = args.early_stopping_patience

	os.makedirs(args.save_dir)

	for epoch in range(args.epochs):

		if patience == 0:
			print("The training will stop because it reaches the limit of patience")
			break

		train_epoch(model, epoch, train_loader, tokenizer, optimizer, args.print_every, device, args.accum_steps)

		if args.early_stopping_criteria == "perplexity":
			loss = evaluate_loss(model, dev_loader, tokenizer, device)
			validation_metric = round(math.exp(loss), 3)
		else:
			validation_metric = evaluate_bleu(model, dev_loader, tokenizer, args.tgt_max_length, args.beam_size, device)

		print(f'Validation {args.early_stopping_criteria}: {validation_metric:.3f}')
		if improved(validation_metric, best_metric, args.early_stopping_criteria):
			print(f'The {args.early_stopping_criteria} improved from {best_metric:.3f} to {validation_metric:.3f}')
			best_metric = validation_metric
			print("Saving checkpoint ...")
			model.save_pretrained(args.save_dir)
			vocab_path = os.path.join(args.save_dir, "vocab")
			if not os.path.exists(vocab_path):
				os.mkdir(vocab_path)
			tokenizer.save_pretrained(vocab_path)
			patience = args.early_stopping_patience
			print("Model saved")
		else:
			patience -= 1
			print(f'Patience ({patience}/{args.early_stopping_patience})')

	print("Loading best checkpoint ...")
	model = T5ForConditionalGeneration.from_pretrained(args.save_dir)#
	model.to(device)
	print("Model was loaded sucessfully.")

	# evaluating dev set
	print("Predicting on dev set ...")
	predictions = predict(model, dev_loader, tokenizer, args.tgt_max_length, args.beam_size, device)
	with open(args.save_dir + "/dev.out", "w") as f:
		for prediction in predictions:
			f.write(prediction.strip() + "\n")

	# evaluating test set
	print("Predicting on test set ...")
	predictions = predict(model, test_loader, tokenizer, args.tgt_max_length, args.beam_size, device)
	with open(args.save_dir + "/test.out", "w") as f:
		for prediction in predictions:
			f.write(prediction.strip() + "\n")


	

