import stanza
import udon2
import pymorphy2
import re
import urllib.request as urr
import urllib.parse as urp
import urllib.error as ure
import sys
from urllib.parse import quote
from bs4 import BeautifulSoup
import random

from typing import List

from process.module import ParaphraseModule
from process.preprocessing_utils import PreprocessingUtils

class LocSkAdjective(ParaphraseModule):
    def __init__(self, name="loc_gen_poss") -> None:
        super().__init__(name=name)
        self.morph = pymorphy2.MorphAnalyzer()
        self.stanza_pymorphy_correspondence = {"Nom": "nomn", "Acc": "accs", "Gen": "gent", "Dat": "datv", "Ins": "ablt", "Loc": "loc2", "Par": "gen2", 
                                               "Sing": "sing", "Plur": "plur", 
                                               "Masc": "masc", "Fem": "femn", "Neut": "neut"}
    
    def load(self, preproc_utils: PreprocessingUtils) -> None:
        # load any tools as `preproc_utils` attributes
        preproc_utils.stanza_model = stanza.Pipeline('ru', processors='tokenize,pos,lemma,depparse,ner')
        self.loaded = True

    # обработка feats --> преобразование в словарь
    def get_feats_dict(self, word_feats):
        feats_dict = dict()
        feats_lst = word_feats.split('|')
        for pair in feats_lst: 
            key, value = pair.split('=')
            feats_dict[key] = value
        return feats_dict

    # находим самого левого ребенка родителя геог. обеъкта в генетиве 
    def get_left_child(self, loc_parent_id, parsed_sentence): 
        left_edge = None
        left_children = []
        for word in parsed_sentence.words: 
            if word.upos == 'ADJ' and word.head == loc_parent_id and word.id < loc_parent_id: 
                left_children.append(word)
        if len(left_children) != 0: 
            left_edge = left_children[0]
        else: 
            left_edge = [elem for elem in parsed_sentence.words if elem.id == loc_parent_id][0]
        return left_edge

    # функция, которая обращаяется к веб-сервису для преобразования географ. названия в прилагательное
    def get_loc_adjective(self, location): 
        source = "https://ws3.morpher.ru/russian/adjectivize?s=" + quote(location)
        request = urr.Request(source)
        try:
            response = urr.urlopen(request)
        except ure.HTTPError:
            sys.exit()
        contents = response.read()
        soup = BeautifulSoup(contents, "html.parser")
        poss_adjectives = [elem.text for elem in soup.find_all('string') if len(soup.find_all('string')) > 0]
        return poss_adjectives

    def detect_genetive_location(self, sentence, preproc_utils): 
        doc = preproc_utils.stanza_model(sentence)
        parsed_sentence = doc.sentences[0]
        for word in parsed_sentence.words: 
            for entity in parsed_sentence.ents: 
                if entity.type == 'LOC' and word.text == entity.text:
                    if word.feats is not None and word.feats.find('Gen') != -1: 
                        return True
        return False

    def process(self, input_text: str, preproc_utils: PreprocessingUtils) -> str:
        changed_sentence = ''
        doc = preproc_utils.stanza_model(sentence)
        parsed_sentence = doc.sentences[0]
        # преобразовываем предложение с генетивом типа "мэр Москвы" в предложение с прилагательным на -ск- (московский мэр)
        sentence_locations = []
        for entity in parsed_sentence.ents: 
            if entity.type == 'LOC':
                sentence_locations.append(entity.text)
        parents_locs = [] 
        for word in parsed_sentence.words: 
            if word.feats is not None:
                word_feats = get_feats_dict(word.feats)
                for entity in sentence_locations: 
                    entity_lst = entity.split(' ')
                    if word.text in entity_lst and word.deprel == 'nmod' and 'Gen' in word_feats.values(): 
                        word_parent = word.head
                        parents_locs.append((entity, word_parent))
        for i, entity_parent in enumerate(parents_locs): 
            entity, parent_id = entity_parent[0], entity_parent[1]
            # левая граница, куда будем добавлять преобразованный вариант
            left_edge = self.get_left_child(parent_id, parsed_sentence)
            location_adjectives = self.get_loc_adjective(entity)
            parent_node = [word for word in parsed_sentence.words if word.id == parent_id][0]
            parent_feats = self.get_feats_dict(parent_node.feats)
            parent_case = self.stanza_pymorphy_correspondence[parent_feats['Case']]
            parent_number = self.stanza_pymorphy_correspondence[parent_feats['Number']]
            parent_gender = self.stanza_pymorphy_correspondence[parent_feats['Gender']]
            left_edge_location = None
            if len(location_adjectives) != 0: 
                if len(location_adjectives) == 1: 
                    if entity == 'США': 
                        adjectivized_location = 'американский'
                        adjectivized_location_upd = self.morph.parse(adjectivized_location)[0].inflect({parent_gender, parent_number, parent_case}).word
                        left_edge_location = ' '.join([adjectivized_location_upd, left_edge.text.lower()])
                    else: 
                        # просто случайно выбираем вариант 
                        adjectivized_location = location_adjectives[0]
                        adjectivized_location_upd = self.morph.parse(adjectivized_location)[0].inflect({parent_gender, parent_number, parent_case}).word
                        left_edge_location = ' '.join([adjectivized_location_upd, left_edge.text])
                elif len(location_adjectives) > 1:
                    if entity == 'США': 
                        adjectivized_location = 'американский'
                        adjectivized_location_upd = self.morph.parse(adjectivized_location)[0].inflect({parent_gender, parent_number, parent_case}).word
                        left_edge_location = ' '.join([adjectivized_location_upd, left_edge.text.lower()])
                    else: 
                        # просто случайно выбираем вариант 
                        adjectivized_location = random.choice(location_adjectives)
                        adjectivized_location_upd = self.morph.parse(adjectivized_location)[0].inflect({parent_gender, parent_number, parent_case}).word
                        left_edge_location = ' '.join([adjectivized_location_upd, left_edge.text])
                        print(left_edge_location)
            # если мы не смогли построить прилагательные для локаций, то возвращаем, как есть
            elif len(location_adjectives) == 0: 
                changed_sentence = sentence
            # теперь изменяем все предложение 
            if len(parents_locs) == 1: 
                changed_sentence = re.sub(left_edge.text, left_edge_location, sentence)
                changed_sentence = re.sub(entity, '', changed_sentence)
                changed_sentence = re.sub('  ', ' ', changed_sentence)
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
            elif len(parents_locs) > 1 and i == 0:
                changed_sentence = re.sub(left_edge.text, left_edge_location, sentence)
                changed_sentence = re.sub(entity, '', changed_sentence)
                changed_sentence = re.sub('  ', ' ', changed_sentence)
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
            elif len(parents_locs) > 1 and i != 0:
                if changed_sentence == '': 
                    changed_sentence = re.sub(left_edge.text, left_edge_location, sentence)
                    changed_sentence = re.sub(entity, '', changed_sentence)
                    changed_sentence = re.sub('  ', ' ', changed_sentence)
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
                elif changed_sentence != '': 
                    changed_sentence = re.sub(left_edge.text, left_edge_location, changed_sentence)
                    changed_sentence = re.sub(entity, '', changed_sentence)
                    changed_sentence = re.sub('  ', ' ', changed_sentence)
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

        return changed_sentence

    def process_batch(self, inputs: List[str], preproc_utils: PreprocessingUtils) -> List[str]:
        outputs = []
        for input_text in inputs:
            if self.detect_genetive_location(input_text, preproc_utils): 
                paraphrased = self.process(input_text, preproc_utils)
                outputs.append(paraphrased)
            else: 
                outputs.append(input_text)
        return outputs

if __name__ == "__main__":
    print("This module is not callable")
    exit()
