

dep_snts = []
dep_uds = []
t = "test"
dep_path = "dependency_tree/penman/pt_bosque-ud-" + t  
with open(dep_path + ".snt", "r", encoding="utf8") as fout, open(dep_path + ".dep", "r", encoding="utf8") as fin:
	for ud, snt in zip(fin, fout):
		dep_snts.append(snt.strip())
		dep_uds.append(ud.strip())


consts= []
const_path = "constituent_tree/penman/Bosque.80"
with open(const_path + ".snt", "r", encoding="utf8") as fout, open(const_path + ".const", "r", encoding="utf8") as fin:
	for line_in, line_out in zip(fin, fout):
		consts.append((line_in.strip(), line_out.strip()))


with open(const_path + "-" + t + ".const", "w", encoding="utf8") as fin, \
			open(const_path + "-" + t + ".snt",  "w", encoding="utf8") as fout, \
			open(dep_path + ".dep.1", "w", encoding="utf8") as fdin, \
			open(dep_path + ".snt.1",  "w", encoding="utf8") as fdout:
	for const in consts:
		if const[1].strip() in dep_snts:
			index = dep_snts.index(const[1].strip())
			print(len(dep_snts))
			fin.write(const[0].strip() + "\n")
			fout.write(const[1].strip() + "\n")
			fdin.write(dep_uds[index].strip() + "\n")
			fdout.write(const[1].strip() + "\n")
			del dep_snts[index]
			del dep_uds[index]

#print(len(dep_sentences))
#print(len(const_sentences))

#print(len(set(const_sentences) - set(dep_sentences)))
