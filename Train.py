from arguments import get_args
#import TransformerTrainer
import GPT2Trainer
import T5Trainer
import mBartTrainer
import numpy as np
import random

if __name__ == "__main__":
	args = get_args()
	global step

if args.pretrained_model == "gpt2":
	#TransformerTrainer.main(args)
	GPT2Trainer.main(args)
elif args.pretrained_model == "t5":
	T5Trainer.main(args)
elif args.pretrained_model == "mbart":
	mBartTrainer.main(args)
else:
	print("model does not exist!")
