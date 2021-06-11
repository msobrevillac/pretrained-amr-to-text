from nltk import word_tokenize
import re
import argparse
import os

# data arguments
parser = argparse.ArgumentParser(description="Main Arguments")
parser.add_argument(
  '-input', '--input', default='dependency', type=str, required=True, help='Input file')
parser.add_argument(
  '-output-dir','--output_dir', type=str, required=True, default="/content/", help='Output directory')


def count_eq(line):
	res = {}
	for keys in line:
		res[keys] = res.get(keys, 0) + 1

	return res.get("=", 0)


def get_info(line):
	subtree = line.split(":")
	rel = subtree[0]
	parent_node = ':'.join(subtree[1:])
	try:
		first_parenthesis = parent_node.index("(")
		parent = parent_node[:first_parenthesis]
		node = parent_node[first_parenthesis:]
	except:
		parent = parent_node
		node = ""
	if node != "":
		result = re.match(r'\(\'(.*)\'',node)
		if result is None:
			node = ""
		else:
			node = result[1]
	return parent, node, rel


def go_recursive(nodes_by_level, depth, current, next_brother):
	str_tree = ""
	for index, node in enumerate(nodes_by_level[depth]):

		if node[2] > current and node[2] < next_brother:
			if node[1].strip() == "":
				str_tree += " (" + node[0] + " "
				_next = 100
				if index < len(nodes_by_level[depth]) - 1:
					_next = nodes_by_level[depth][index+1][2]
	
				if depth < len(nodes_by_level) - 1:
					str_tree += go_recursive(nodes_by_level, depth+1, node[2], _next)
				str_tree += ")"

			else:
				str_tree += "(" + node[0] + " " + node[1] + ") "

	return " ".join(str_tree.split())
	


def build_constituent_tree(lines):

	nodes_by_level = {}
	for index, line in enumerate(lines[1:]):
		if not line.startswith("<"):
			if ":" in line:
				parent, node, rel = get_info(line)
				depth = count_eq(rel)
				if depth not in nodes_by_level:
					nodes_by_level[depth] = [(parent, node, index)]
				else:
					nodes_by_level[depth] = nodes_by_level[depth] + [(parent, node, index)]

	str_tree = go_recursive(nodes_by_level, 0, -1, 100)
	return "(snt " + str_tree.strip() + ")"


def read_constituency_files(fname, output_dir):
	with open(fname, "r", encoding="ISO-8859-1") as f:
		name = os.path.basename(fname).split(".")
		if len(name) > 1:
			name = ".".join(name[:-1])
		else:
			name = name[0]
		fsent = open(os.path.join(output_dir, name+".snt"), "w", encoding="utf8")
		fconst = open(os.path.join(output_dir, name+".const"), "w", encoding="utf8")

		parts = f.read().split("\n\n")
		for index, part in enumerate(parts):
			lines = part.split("\n")
			if not lines[0].startswith("SOURCE:"):
				continue
			sentence = ' '.join(lines[1].split()[1:])
			if len(word_tokenize(sentence)) <= 23:
				tree = build_constituent_tree(lines[2:])
				fsent.write(' '.join(word_tokenize(sentence)).lower().strip() + "\n")
				fconst.write(tree.lower().strip() + "\n")
		fsent.close()
		fconst.close()


if __name__ == "__main__":
	args = parser.parse_args()

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	read_constituency_files(args.input, args.output_dir)


		
