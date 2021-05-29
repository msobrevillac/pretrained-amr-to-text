import torch
import argparse
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelWithLMHead

from constants import SPECIAL_TOKENS, AMR_SPECIAL_TOKENS

from amr_utils import get_data_loaders
from utils import trim_batch, apply_loss, set_seed, improved
import sacrebleu
import os


def decode(labels, tokenizer):
	text = tokenizer.decode(labels, clean_up_tokenization_spaces=False)
	generate_index = text.rfind("<generate>")
	if generate_index != -1:
		generate_index += len("<generate>")
	eos_index = text.rfind("<eos>")        

	if generate_index != -1 and eos_index != -1:
		cleaned_text = text[generate_index:eos_index]
	elif generate_index == -1 and eos_index != -1:
		cleaned_text = text[:eos_index]
	elif generate_index != -1 and eos_index == -1:
		cleaned_text = text[generate_index:]
	else:  		        				        
		cleaned_text = text
	return cleaned_text


def train_epoch(model, epoch, loader, optimizer, print_every, device, pad=-100, accumulation_steps):#, pad=-100:
	model.train()
	total_loss = 0
	n = 0
	for index, batch in enumerate(loader, 0):
		batch = trim_batch(batch, pad)
   
		input_ids, lm_labels, token_type_ids, attention_mask, _, _, _, _ =\
			tuple(input_tensor.to(device) for input_tensor in batch)
                
		outputs = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids)
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


def evaluate_loss(model, loader, device, pad=-100):#, pad=-100:
	model.eval()
	total_loss = 0
	n = 0
	with torch.no_grad():
		for index, batch in enumerate(loader, 0):
			batch = trim_batch(batch, pad)
   
			input_ids, lm_labels, token_type_ids, attention_mask, _, _, _, _ =\
				tuple(input_tensor.to(device) for input_tensor in batch)
                
			loss_ce = model(input_ids=input_ids, labels=lm_labels, token_type_ids=token_type_ids)
			loss_ce = loss_ce[0]
			total_loss += loss_ce.item()
			n += 1

	return total_loss / n
	#print(f'Validation Loss: {total_loss/n}')


def evaluate_bleu(model, tokenizer, loader, device, max_length=180, beam_size=15):

	bos, eos, ctx, ans, que, pad, gen = \
		tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

	references = []
	hypothesis = [] 
	model.eval()
	with torch.no_grad():
		for index, batch in enumerate(loader, 0):
			batch = trim_batch(batch, pad)

			_, lm_labels, _, _,input_ids, _, token_type_ids, attention_mask =\
						tuple(input_tensor.to(device) for input_tensor in batch)

			sample_outputs = model.generate(input_ids,
								pad_token_id=pad,
								max_length=max_length,
								num_beams=beam_size,
								early_stopping=True,
								no_repeat_ngram_size=2,
								repetition_penalty=2.5,
								num_return_sequences=1,
								eos_token_id=eos)

			# reconstruct reference   
			for i, lm_label in enumerate(lm_labels):
				labels = [e.item() for e in lm_label if e != -100]     
				reference_text = decode(labels, tokenizer)
				references.append(reference_text)      

			# generated output sequence   
			for i, sample_output in enumerate(sample_outputs):
				output_text = decode(sample_output.tolist(), tokenizer)
				hypothesis.append(output_text)

	bleu = sacrebleu.corpus_bleu(hypothesis, [references])
	return bleu.score   


def predict(model, tokenizer, loader, device, pad=-100, eos=5506, max_length=180, beam_size=15, out=None):

	model.eval()
	for index, batch in enumerate(loader, 0):
		batch = trim_batch(batch, pad)

		_, _, _, _,input_ids, _, token_type_ids, attention_mask =\
					tuple(input_tensor.to(device) for input_tensor in batch)

		sample_outputs = model.generate(input_ids,
								pad_token_id=pad,
								max_length=max_length,
								num_beams=beam_size,
								early_stopping=True,
								no_repeat_ngram_size=2,
								repetition_penalty=2.5,
								num_return_sequences=1,
								eos_token_id=eos)

		# generated sequence
		for i, sample_output in enumerate(sample_outputs):
			output = tokenizer.decode(sample_output.tolist(), clean_up_tokenization_spaces=False)

			generate_index = output.rfind("<generate>")
			if generate_index != -1:
				generate_index += len("<generate>")
			eos_index = output.rfind("<eos>")        

			if generate_index != -1 and eos_index != -1:
				output_text = output[generate_index:eos_index]
			elif generate_index == -1 and eos_index != -1:
				output_text = output[:eos_index]
			elif generate_index != -1 and eos_index == -1:
				output_text = output[gnerate_index:]
			else:  		        				        
				output_text = output
			if out is not None:
				out.write(output_text.strip() + "\n")


def main(args):

	set_seed(args.seed)

	if args.parsing:
		generation = False
	else:
		generation = True


	tokenizer = AutoTokenizer.from_pretrained(args.model)
	tokenizer.model_max_length=args.src_max_length#1024  
	tokenizer.sep_token = '<sep>'

	tokenizer.add_tokens(SPECIAL_TOKENS)

	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	if args.representation == "amr":
		tokenizer.add_tokens(AMR_SPECIAL_TOKENS)

	model = AutoModelWithLMHead.from_pretrained(args.model)#
	model.to(device)

	train_dataloader = get_data_loaders(args.train_source, tokenizer, model, tgt_path=args.train_target, batch_size=args.batch_size, max_length=args.src_max_length, generation=generation)


	dev_dataloader = get_data_loaders(args.dev_source, tokenizer, model,
			tgt_path = args.dev_target, shuffle=False, is_train=False,
			batch_size=args.batch_size, max_length=args.src_max_length, generation=generation)

	if args.fixed_embeddings:
		fixed_name = "wte"
		for name, param in model.named_parameters():
			if fixed_name in name:
				param.requires_grad = False
				print("Freezing ", fixed_name)


	optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

	bos, eos, ctx, ans, que, pad, gen = \
		tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

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

		train_epoch(model, epoch, train_dataloader, optimizer, args.print_every, device, pad=pad, args.accum_steps)

		if args.early_stopping_criteria == "perplexity":
			loss = evaluate_loss(model, dev_dataloader, device, pad=pad)
			validation_metric = round(math.exp(loss), 3)
		else:
			validation_metric = evaluate_bleu(model, tokenizer, dev_dataloader, device, \
				max_length=args.tgt_max_length, beam_size=args.beam_size)

		print(f'Validation {args.early_stopping_criteria}: {validation_metric:.3f}')
		if improved(validation_metric, best_metric, args.early_stopping_criteria):
			print(f'The {args.early_stopping_criteria} improved from {best_metric:.3f} to {validation_metric:.3f}')
			best_metric = validation_metric
			print("Saving checkpoint ...")
			model.save_pretrained(args.save_dir)
			#torch.save(model.state_dict(), args.save_dir + "model.bin")
			if not os.path.exists(args.save_dir + "vocab"):
				os.mkdir(args.save_dir + "vocab")
			#tokenizer.save_vocabulary(args.save_dir + "vocab")
			tokenizer.save_pretrained(args.save_dir + "vocab")
			patience = args.early_stopping_patience
			print("Model saved")
		else:
			patience -= 1
			print(f'Patience ({patience}/{args.early_stopping_patience})')

	print("Loading best checkpoint ...")
	model = AutoModelWithLMHead.from_pretrained(args.save_dir)#
	model.to(device)
	print("Model was loaded sucessfully.")


	if args.dev_source is not None and args.dev_target is not None:
		dataloader = get_data_loaders(args.dev_source, tokenizer, model,
				tgt_path = args.dev_target, shuffle=False, is_train=False,
				batch_size=args.batch_size, max_length=args.src_max_length, generation=generation)
		f = open(args.save_dir + "dev.out", "w")
		predict(model, tokenizer, dataloader, device, pad=pad, eos=eos, max_length=args.tgt_max_length, beam_size=args.beam_size, out=f)
		f.close()

	if args.test_source is not None:
		dataloader = get_data_loaders(args.test_source, tokenizer, model,
							shuffle=False, is_train=False, batch_size=args.batch_size,
							max_length=args.src_max_length, generation=generation)
		f = open(args.save_dir + "test.out", "w")
		predict(model, tokenizer, dataloader, device, pad=pad, eos=eos, max_length=args.tgt_max_length, beam_size=args.beam_size, out=f)
		f.close()
		


