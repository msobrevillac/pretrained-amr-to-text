from nltk import word_tokenize
import re
import argparse
import os
from os import listdir
from conllu import parse_incr

# data arguments
parser = argparse.ArgumentParser(description="Main Arguments")
parser.add_argument(
  '-input', '--input', default='dependency', type=str, required=True, help='Input file')
parser.add_argument(
  '-output-dir','--output_dir', type=str, required=True, default="/content/", help='Output directory')



valid_feats = ['Tense', 'ClauseType', 'Number', 'Definite', 'PronType', 'Gender', 'Person', 'Poss', 'Degree', 'Voice', 'NumType', 'Mood', 'Foreign', 'Aspect']


def process_feats(feats_dict):
	str = ""
	if feats_dict is None:
		return ""
	for feat, value in feats_dict.items():
		if feat in valid_feats:
			str += " :" + feat.lower() + " " + value.lower()
	return str.strip()


def process(tree, root=True):

	if tree.token['upos'] == "PUNCT":
		return ""

	lemma = tree.token['lemma']
	linearised_tree = "(" + lemma.lower()

	if not root:
		rel = " :" + tree.token['deprel'].lower()
		linearised_tree = rel + " " + linearised_tree
	
	if len(tree.children) > 0:
		for child in tree.children:
			linearised_tree += " " + process(child, root=False)
	linearised_tree += ")"

	return linearised_tree.strip()


def read_dependency_files(fname, output_dir):
	with open(fname, "r", encoding="utf8") as f:
		name = os.path.basename(fname).split(".")
		if len(name) > 1:
			name = ".".join(name[:-1])
		else:
			name = name[0]
		fsent = open(os.path.join(output_dir, name+".snt"), "w", encoding="utf8")
		fdep = open(os.path.join(output_dir, name+".dep"), "w", encoding="utf8")

		for tokenlist in parse_incr(f):

			if len(word_tokenize(tokenlist.metadata['text'])) <= 23:

				output = process(tokenlist.to_tree())
				fsent.write((' '.join(word_tokenize(tokenlist.metadata['text']))).lower().strip() + "\n")
				fdep.write(output.strip() + "\n")

		fsent.close()
		fdep.close()

if __name__ == "__main__":
	args = parser.parse_args()

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	read_dependency_files(args.input, args.output_dir)


