import os
import json
import jsonpickle
from typing import List
from tqdm import tqdm
from dynaconf import settings
from nltk import download

# download("stopwords")
from nltk.corpus import stopwords

class Data:

    def __init__(self, data_dir="data/wiktionary/", out_of_vocabulary = [], source_data = "annotated_sample_wiktionary.json"):
        self.check_data(data_dir, os.path.join(settings["data_path"], source_data), "train")
        self.train_data = self.load_data(data_dir, out_of_vocabulary, "train")
        self.data = self.train_data
        self.entities = self.get_entities(self.data)
        self.train_relations = self.get_relations(self.train_data)
        self.relations = self.train_relations

    def check_data(self, data_dir, source_data, data_type = "train"):
        #verify that triples are present, otherwise generate triples from source_data
        #check path
        if os.path.exists("./"+data_dir+"/"+data_type+".txt"):
            return
        #create training dataset from annotated souce_data
        in_file = open(source_data, "r")
        json_string = json.load(in_file)
        annotated_corpus = jsonpickle.decode(json_string)
        #create triples
        out_file = open("%s%s.txt" % (data_dir, data_type), "w", encoding="utf-8")
        print("Creating training data from", source_data, "to", "%s%s.txt" % (data_dir, data_type))
        for definition in tqdm(annotated_corpus):
            definiendum = definition.annotations["definiendum"]
            for token in definition.tokens:
                if not "dsr" in token.annotations or not "lemma" in token.annotations or token.annotations["dsr"] == "O" or token.surface in stopwords.words("english"):
                    continue
                print(definiendum.lower()+"\t"+token.annotations["dsr"]+"\t"+token.annotations["lemma"].lower(), file=out_file)
        out_file.close()

    def load_data(self, data_dir, out_of_vocabulary = [], data_type="train"):
        with open("%s%s.txt" % (data_dir, data_type), "r", encoding="utf-8") as f:
            data = f.read().strip().split("\n")
            data = [i.split("\t") for i in data if i and i.split("\t")[0] not in out_of_vocabulary and i.split("\t")[2] not in out_of_vocabulary]
            data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data) -> List[str]:
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data) -> List[str]:
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities
