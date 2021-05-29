from conllu import parse_incr
from nltk.tokenize import word_tokenize
from os import listdir


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
	form = tree.token['form']
	str_feats = process_feats(tree.token['feats'])
	linearised_tree = "( " + form 
	if str_feats.strip() != "":
		linearised_tree += " " + str_feats

	if not root:
		rel = " :" + tree.token['deprel'].lower()
		linearised_tree = rel + " " + linearised_tree
	
	if len(tree.children) > 0:
		for child in tree.children:
			linearised_tree += " " + process(child, root=False)
	linearised_tree += " )"

	return linearised_tree.strip()

path = "data/deep-ud/en/test+/"

names = []
for fname in listdir(path):
	fname = fname.replace(".conllu", "").replace("_DEEP", "")
	if fname not in names:
		names.append(fname)

#print(names)
'''
fsource = open(path + "train_src", "w")
ftarget = open(path + "train_tgt", "w")

for name in names:
	print(name)
	fin = open(path + name + "_DEEP.conllu", "r")
	fout = open(path + name + ".conllu", "r")
	for tokenlist in parse_incr(fin):
		print(tokenlist)
		fsource.write(process(tokenlist.to_tree()).strip() + "\n")

	for tokenlist in parse_incr(fout):
		ftarget.write((' '.join(word_tokenize(tokenlist.metadata['text']))) + "\n")

fsource.close()
ftarget.close()
'''

for name in names:
	print(name)
	fsource = open(path + name  + ".src", "w")
	fin = open(path + name + "_DEEP.conllu", "r")
	for tokenlist in parse_incr(fin):
		fsource.write(process(tokenlist.to_tree()).strip() + "\n")

	fsource.close()


