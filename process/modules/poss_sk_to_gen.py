import stanza 
import udon2
import pymorphy2
import re

from typing import List

from process.module import ParaphraseModule
from process.preprocessing_utils import PreprocessingUtils

class PossSkAdjToGenetiveModule(ParaphraseModule):
    def __init__(self, name="poss_sk_to_gen") -> None:
        super().__init__(name=name)
        self.morph = pymorphy2.MorphAnalyzer()
        self.adjective_dict_path = "modules/poss_sk_adjectives_dict.json"
        self.stanza_pymorphy_correspondence = {"Nom": "nomn", "Acc": "accs", "Gen": "gent", "Dat": "datv", "Ins": "ablt", "Loc": "loc2", "Par": "gen2", 
                                               "Sing": "sing", "Plur": "plur", 
                                               "Masc": "masc", "Fem": "femn", "Neut": "neut"}
    
    def load(self, preproc_utils: PreprocessingUtils) -> None:
        # load any tools as `preproc_utils` attributes
        preproc_utils.stanza_model = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse,ner')
        self.loaded = True

    def get_adjectives_dict(self, file_path): 
        adjectives_dictionary = None
        with open(file_path, 'r') as fin:
            adjectives_data = json.load(fin)
            adjectives_dictionary = verbs_data['sk adjectives']
        return adjectives_dictionary
    
    def detect_sk_adj_sentence(self, sentence, preproc_utils): 
    doc = preproc_utils.stanza_model(sentence)
    parsed_sentence = doc.sentences[0]
    for word in parsed_sentence.words: 
        sk_adg_pattern = re.compile('[а-яА-Я]+ск(ий|ому|ого|им|ом|ая|ой|ую|ою|ие|их|им|ими)')
        if word.upos == 'ADJ' and re.fullmatch(sk_adg_pattern, word.text):
            return True
    return False

    # обработка feats --> преобразование в словарь
    def get_feats_dict(self, word_feats):
        feats_dict = dict()
        feats_lst = word_feats.split('|')
        for pair in feats_lst: 
            key, value = pair.split('=')
            feats_dict[key] = value
        return feats_dict

    # функция, которая находит прилагательное с суффиксом -ск- 
    # на вход подаем предложение после парсинга Stanza
    # важно: очень сложно учесть все исключения, поэтому пока находит только одно прилагательное
    def find_sk_adjective(self, parsed_sentence): 
        sk_adjective = None
        for word in parsed_sentence.words: 
            sk_adg_pattern = re.compile('[а-яА-Я]+ск(ий|ому|ого|им|ом|ая|ой|ую|ою|ие|их|им|ими)')
            if word.upos == 'ADJ' and re.fullmatch(sk_adg_pattern, word.text): 
                sk_adjective = word
        return sk_adjective
    
    # преобразование предложения с прилагательным на -ск- ("московский мэр") в предложение с генетивным поссором ("мэр Москвы")
    def process(self, input_text: str, adjectives_dict: dict, preproc_utils: PreprocessingUtils) -> str:
        changed_sentence = ''
        doc = preproc_utils.stanza_model(input_text)
        parsed_sentence = doc.sentences[0]
        sk_adjective = self.find_sk_adjective(parsed_sentence)
        adjective_parent_idx = sk_adjective.head
        adjective_parent = [elem for elem in parsed_sentence.words if elem.id==adjective_parent_idx][0]
        parent_feats = self.get_feats_dict(adjective_parent.feats)
        if sk_adjective.lemma in adjectives_dict.keys(): 
            location = adjectives_dict[sk_adjective.lemma]
            location_gen = self.morph.parse(location)[0].inflect({'gent'}).word
            # gender, number, case
            if location_gen == 'сша':
                parent_adjectivized = ' '.join([adjective_parent.text, location_gen])
                changed_sentence = re.sub(sk_adjective.text, '', input_text)
                changed_sentence = re.sub(adjective_parent.text, parent_adjectivized, changed_sentence)
                changed_sentence = re.sub('  ', ' ', changed_sentence)
                changed_sentence = re.sub(location_gen, location_gen.upper(), changed_sentence)
                sentence_words = changed_sentence.split(' ')
                if sentence_words[0] == '': 
                    sentence_words.pop(0)
                    for i, word in enumerate(sentence_words): 
                        if i != 0 and word.istitle(): 
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].lower(), changed_sentence)
                        elif i == 0 and word.islower():
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].title(), changed_sentence)
                else: 
                    for i, word in enumerate(sentence_words): 
                        if i != 0 and word.istitle(): 
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].lower(), changed_sentence)
                        elif i == 0 and word.islower():
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].title(), changed_sentence)
            else: 
                parent_adjectivized = ' '.join([adjective_parent.text, location_gen])
                changed_sentence = re.sub(sk_adjective.text, '', input_text)
                changed_sentence = re.sub(adjective_parent.text, parent_adjectivized, changed_sentence)
                changed_sentence = re.sub('  ', ' ', changed_sentence)
                changed_sentence = re.sub(location_gen, location_gen.title(), changed_sentence)
                sentence_words = changed_sentence.split(' ')
                if sentence_words[0] == '': 
                    sentence_words.pop(0)
                    for i, word in enumerate(sentence_words): 
                        if i != 0 and word.istitle(): 
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].lower(), changed_sentence)
                        elif i == 0 and word.islower():
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].title(), changed_sentence)
                else: 
                    for i, word in enumerate(sentence_words): 
                        if i != 0 and word.istitle(): 
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].lower(), changed_sentence)
                        elif i == 0 and word.islower():
                            changed_sentence = re.sub(sentence_words[i], sentence_words[i].title(), changed_sentence)
        else: 
            changed_sentence = input_text
        return changed_sentence

    def process_batch(self, inputs: List[str], preproc_utils: PreprocessingUtils) -> List[str]:
        adjectives_dict = self.get_adjectives_dict(self.adjective_dict_path)
        outputs = []
        for input_text in inputs:
            if self.detect_sk_adj_sentence(input_text, preproc_utils): 
                paraphrased = self.process(input_text, adjectives_dict, preproc_utils)
                outputs.append(paraphrased)
            else: 
                outputs.append(input_text)
        return outputs

if __name__ == "__main__":
    print("This module is not callable")
    exit()
