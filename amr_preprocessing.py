import penman
from utils import parse_corpus
import argparse
import os

parser = argparse.ArgumentParser(description="Pre processing")
parser.add_argument(
	'-input','--input', type=str, required=True, help='Input folder')
parser.add_argument(
	'-output', '--output', type=str, required=True, help='Output folder')
parser.add_argument(
	'-mode', '--mode', type=str,  choices=['penman', 'penman-frames', 'dfs', 'dfs-relations', 'dfs-frames', 'dfs-relations-frames'], default="penman", required=True, help='mode')
parser.add_argument(
	'-no-variable', '--no-variable', action='store_true', required=False, help='No variables. This only works for penman notation.')
parser.add_argument(
  '-parsing','--parsing', action='store_true', required=False, help='Parsing or generation')

def remove_wiki(g):
	#removing wiki relations
	triples = []
	for	index, triple in enumerate(g.triples):
		if triple[1].strip() == ":wiki": #always remove :wiki relations
			continue
		triples.append(triple)
	g.triples = triples


def get_recursive_dfs(node):
	tokens = []
	for item in node[1]:
		if type(item[1]) == tuple:
			tokens.append(item[0])
			tokens += get_recursive_dfs(item[1])
		else:
			if item[0] != "/":
				tokens.append(item[0])
			tokens.append(item[1])
	return tokens

def get_dfs(amr, relations=True, frames=True, parsing=False):

	g = penman.decode(amr)
	g.epidata = {}

	remove_wiki(g)

	variables = {instance[0]:instance[2] for instance in g.instances()}
	t = penman.parse(penman.encode(g))
	var, branch = t.node
	tokens = get_recursive_dfs(t.node)
	tokens = [variables[token] if token in variables else token for token in tokens]

	if not relations:
		tokens = [token.lower() for token in tokens if not token.startswith(":")]
	if not frames:
		aux = []
		for token in tokens:
			if len(token.split("-")) > 1:
				try:
					nSplits = token.split("-")
					number = int(nSplits[len(nSplits)-1])
					aux.append(('-'.join(nSplits[:len(nSplits)-1])).lower())
				except:
					aux.append(token.lower())
			else:
				aux.append(token.lower())
		tokens = aux

	if parsing:
		return (' '.join(tokens)).lower().strip()

	return (' '.join(tokens)).lower().strip().replace("\"","")


def get_penman(amr, frames=True, parsing=False, no_variable=False):

	g = penman.decode(amr)
	g.epidata = {}

	remove_wiki(g)

	if not frames:
		for	index, triple in enumerate(g.triples):
			splits = triple[2].split("-")
			if len(splits) > 1:
				try:
					number = int(splits[len(splits)-1])
					frame = '-'.join(splits[:len(splits)-1])
					g.triples[index] = (triple[0], triple[1], frame) 
				except:
					continue
	aux_amr = penman.encode(g)
	str_amr = ""

	instances = {instance[0]:instance[2] for instance in g.instances()}

	tokens = aux_amr.split()

	for index, token in enumerate(tokens):		
		if token.strip() != "":
			aux = token.replace("(", "").replace(")","")
			if aux in instances:
				if index+1 < len(tokens) and tokens[index+1] == "/":
					if no_variable:
						str_amr += token.replace(aux, "")
						tokens[index+1] = ""
					else:
						str_amr += token + " "
				else:
					t = token.replace(aux, "")		
					str_amr += instances[aux] + "-ref" + t + " "
			else:
				str_amr += token + " "

	if parsing:
		return str_amr.strip().lower()

	return str_amr.strip().lower().replace("\"","")


def process(args, amrs, dataset):
	output_dir = os.path.join(args.output, dataset)
	os.makedirs(output_dir)

	fout_src = open(output_dir + "/amr.txt", "w", encoding="utf8")
	fout_tgt = open(output_dir + "/sentence.txt", "w", encoding="utf8")

	for index, amr in enumerate(amrs):
		str_input = ""
		if args.mode == "penman":
			str_input = get_penman(amr['amr'], parsing=args.parsing, no_variable=args.no_variable)
		elif args.mode == "penman-frames":
			str_input = get_penman(amr['amr'], frames=False, parsing=args.parsing, no_variable=args.no_variable)
		elif args.mode == "dfs":
			str_input = get_dfs(amr['amr'], parsing=args.parsing)
		elif args.mode == "dfs-relations":
			str_input = get_dfs(amr['amr'], relations=False, parsing=args.parsing)
		elif args.mode == "dfs-frames":
			str_input = get_dfs(amr['amr'], frames=False, parsing=args.parsing)
		else:
			str_input = get_dfs(amr['amr'], relations=False, frames=False, parsing=args.parsing)
		fout_src.write(str_input + "\n")
		fout_tgt.write(amr['sentence_pt'].strip().lower() + "\n")

	fout_src.close()
	fout_tgt.close()

if __name__ == "__main__":
	args = parser.parse_args()

	path_train = args.input + "/train/train.txt"
	path_dev = args.input + "/dev/dev.txt"
	path_test = args.input + "/test/test.txt"

	amr_train = parse_corpus(path_train)
	amr_dev = parse_corpus(path_dev)
	amr_test = parse_corpus(path_test)

	print("Loading ...", len(amr_train))
	print("Loading ...", len(amr_dev))
	print("Loading ...", len(amr_test))

	process(args, amr_train, "train")
	process(args, amr_dev, "dev")
	process(args, amr_test, "test")


