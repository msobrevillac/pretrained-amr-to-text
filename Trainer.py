from Dataloader import WebnlgDataset, process_data
from util import set_seed
import torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration

def evaluate(model, loader, tokenizer, max_length, beam_size, print_every, device):
	model.eval()
	predictions = []
	with torch.no_grad():
		for _, data in enumerate(loader, 0):
			ids = data['source_ids'].to(device, dtype = torch.long)
			mask = data['source_mask'].to(device, dtype = torch.long)

			generated_ids = model.generate(
				input_ids = ids,
				attention_mask = mask, 
				max_length=max_length, 
				num_beams=beam_size,
				repetition_penalty=2.5, 
				length_penalty=1.0, 
				early_stopping=True)

			preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

			if _%print_every==0:
				print(f'Completed {_}')

			predictions.extend(preds)
	return predictions



def train_epoch(model, epoch, loader, tokenizer, optimizer, print_every, device):
	model.train()
	for _,data in enumerate(loader, 0):
		y = data['target_ids'].to(device, dtype = torch.long)
		y_ids = y[:, :-1].contiguous()
		lm_labels = y[:, 1:].clone().detach()
		lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
		ids = data['source_ids'].to(device, dtype = torch.long)
		mask = data['source_mask'].to(device, dtype = torch.long)

		outputs = model(input_ids = ids, attention_mask = mask, decoder_input_ids=y_ids, lm_labels=lm_labels)
		loss = outputs[0]
        
		if _%print_every==0:
			print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


def train(args):

	set_seed(args.seed)

	device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')

	if len(args.train_source) != len(args.train_target):
		print("Error.Number of inputs in train are not the same")
		return

	df_train = process_data(args.train_source[0], args.train_target[0])
	df_dev = process_data(args.dev_source[0])
	df_test = process_data(args.test_source[0])


	tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

	train_set = WebnlgDataset(df_train, tokenizer, args.max_src_length, args.max_tgt_length)
	dev_set = WebnlgDataset(df_dev, tokenizer, args.max_src_length, args.max_tgt_length)
	test_set = WebnlgDataset(df_test, tokenizer, args.max_src_length, args.max_tgt_length)

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


	model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
	model = model.to(device)

	#if args.optimizer == "adam":
	optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)

	for epoch in range(args.epochs):
		train_epoch(model, epoch, train_loader, tokenizer, optimizer, args.print_every, device)

	# evaluating dev set
	predictions = evaluate(model, dev_loader, tokenizer, args.max_tgt_length, args.beam_size, args.print_every, device)
	with open(args.save_dir + "/dev.out", "w") as f:
		for prediction in predictions:
			f.write(prediction.strip() + "\n")

	# evaluating test set
	predictions = evaluate(model, test_loader, tokenizer, args.max_tgt_length, args.beam_size, args.print_every, device)
	with open(args.save_dir + "/test.out", "w") as f:
		for prediction in predictions:
			f.write(prediction.strip() + "\n")


	

