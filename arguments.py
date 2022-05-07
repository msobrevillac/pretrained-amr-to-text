""" File to hold arguments """
import argparse

# data arguments

parser = argparse.ArgumentParser(description="Main Arguments")

parser.add_argument(
  '-train-src', '--train_source', type=str, required=False, help='Path to train source dataset')
parser.add_argument(
  '-train-tgt', '--train_target', type=str, required=False, help='Path to train target dataset')

parser.add_argument(
  '-dev-src', '--dev_source', type=str,  required=False, help='Path to dev source dataset')
parser.add_argument(
  '-dev-tgt', '--dev_target', type=str, required=False, help='Path to dev target dataset')

parser.add_argument(
  '-test-src', '--test_source', type=str, default="", required=False, help='Path to test source dataset')
parser.add_argument(
  '-test-tgt', '--test_target', type=str, default="", required=False, help='Path to test target dataset')

# training parameters
parser.add_argument(
  '-epochs', '--epochs', type=int, required=False, default=3, help='Number of training epochs')
parser.add_argument(
  '-print-every', '--print_every', type=int, default=500, required=False, help='Print the loss/ppl every training steps')

parser.add_argument(
  '-batch-size', '--batch_size', type=int, required=False, default=32, help='Batch size')
parser.add_argument(
  '-src-max-length', '--src_max_length', type=int, required=False, default=180, help='Max length in encoder')
parser.add_argument(
  '-tgt-max-length', '--tgt_max_length', type=int, required=False, default=80, help='Max length in decoder')


# hyper-parameters
parser.add_argument(
  '-optimizer','--optimizer', type=str, required=False, default="AdamW", help='Optimizer that will be used')
parser.add_argument(
  '-lr','--learning_rate', type=float, required=False, default=0.0001, help='Learning rate')
parser.add_argument(
  '-adam-epsilon','--adam_epsilon', type=float, default=1.0e-8, required=False, help='Adam epsilon')


parser.add_argument(
  '-accum-steps','--accum_steps', type=int, required=False, default=1, help='Gradient Accumulation')

parser.add_argument(
  '-beam-size','--beam_size', type=int, required=False, default=5, help='Beam search size ')
parser.add_argument(
  '-beam-alpha', '--beam_alpha', type=float, required=False, default=0.2, help='Alpha value for Beam search')

parser.add_argument(
  '-seed', '--seed', type=int, required=False, help='Seed')
parser.add_argument(
  '-gpu','--gpu', action='store_true', required=False, help='Use GPU or CPU')

parser.add_argument(
  '-fixed-embed','--fixed_embeddings', action='store_true', required=False, help='Use GPU or CPU')

parser.add_argument(
  '-parsing','--parsing', action='store_true', required=False, help='Parsing or generation')

parser.add_argument(
  '-save-dir','--save_dir', type=str, required=True, default="/content/", help='Output directory')

parser.add_argument(
  '-representation', '--representation', type=str, default="deep-ud", required=False, help='Kind of representation')

parser.add_argument(
  '-task', '--task', type=str, default="lemma", choices= ['ud', 'cg', 'graph', 'lemma'], required=True, help='Task. Default: lemma (lemma-to-text generation)')

parser.add_argument(
  '-model','--model', type=str, required=False, default="pierreguillou/gpt2-small-portuguese", help='Path for a pre-trained model file (just to perform transfer learning)')

parser.add_argument(
  '-pretrained-model', '--pretrained-model', default='gpt2', type=str, choices=['gpt2', 'bart', 'mbart', 't5', 'mt5', "t5-multi"], required=False, help='Pretrained model to be used')

parser.add_argument(
  '-early-stopping-patience','--early-stopping-patience', type=int, default=-1, required=False, help='Early stopping patience')

parser.add_argument(
  '-eval-criteria','--eval-criteria', type=str, default="perplexity", choices=['perplexity', 'bleu'], help='Criteria to evaluate (perplexity|bleu)')


def get_args():
  args = parser.parse_args()
  return args


