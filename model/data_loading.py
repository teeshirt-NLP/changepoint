import random
from utils import load_obj


class DataLoader:
    def __init__(self, RUNTIME_SETTINGS):
        self.data_path = RUNTIME_SETTINGS['data_path']
        self.paragraphs, self.tokenizer, self.detokenizer, self.vocabulary_size = self.load_data()

    def load_data(self):
        paragraphs = load_obj(self.data_path + "/" + "Paragraphdata-training")
        vocabulary = load_obj(self.data_path + "/" + "BOCE.English.400K.vocab")
        
        detokenizer = dict(enumerate(vocabulary))
        tokenizer = dict(zip(detokenizer.values(), detokenizer.keys()))
        return paragraphs, tokenizer, detokenizer, len(vocabulary)

    def encode_sentences(self, sentences):
        encoded_sentences = []
        words_lists = [sentence.split(" ") for sentence in sentences]

        for words in words_lists:
            non_empty_words = [word for word in words if word]
            encoded_words = [self.tokenizer[word] for word in non_empty_words if word in self.tokenizer.keys()]
            encoded_sentences.append(encoded_words)

        return encoded_sentences

    def generate_single_doublet(self):
        span = len(self.paragraphs)
        target1, target2 = random.sample(range(span), 2)

        s1, s2 = self.paragraphs[target1], self.paragraphs[target2]
        v1, v2 = self.encode_sentences(s1), self.encode_sentences(s2)

        v1, v2 = [x for x in v1 if len(x) > 2], [x for x in v2 if len(x) > 2]

        if len(v1) > 2:
            v1_index = random.choice(range(len(v1) - 2))
            v1 = v1[v1_index:v1_index + 2]

        if len(v2) > 2:
            v2_index = random.choice(range(len(v2) - 2))
            v2 = v2[v2_index:v2_index + 2]

        return [v1, v2]


    def generate_single_doublet_padded(self, max_nwords=100):
        span = len(self.paragraphs)
        target1, target2 = random.sample(range(span), 2)

        s1, s2 = self.paragraphs[target1], self.paragraphs[target2]
        v1, v2 = self.encode_sentences(s1), self.encode_sentences(s2)

        v1, v2 = [x for x in v1 if len(x) > 2], [x for x in v2 if len(x) > 2]

        if len(v1) > 2:
            v1_index = random.choice(range(len(v1) - 2))
            v1 = v1[v1_index:v1_index + 2]

        if len(v2) > 2:
            v2_index = random.choice(range(len(v2) - 2))
            v2 = v2[v2_index:v2_index + 2]

        # Padding
        padding_value = int(1e6)
        v1 = [a + [padding_value] * (max_nwords - len(a)) for a in v1]
        v2 = [a + [padding_value] * (max_nwords - len(a)) for a in v2]

        return [v1, v2]


    def generate_batch(self, n, max_nwords=100):
        batch = []
        while len(batch) < n:
            doublet = self.generate_single_doublet_padded(max_nwords)
            if all(len(sublist) == 2 and len(sublist[0]) == max_nwords and len(sublist[1]) == max_nwords for sublist in doublet):
                batch.append(doublet)
        return [batch]


    def generate_single_doublet_raw(self):
        span = len(self.paragraphs)
        target1, target2 = random.sample(range(span), 2)

        s1, s2 = self.paragraphs[target1], self.paragraphs[target2]

        s1, s2 = [x for x in s1 if x], [x for x in s2 if x]

        if len(s1) > 2:
            s1_index = random.choice(range(len(s1) - 2))
            s1 = s1[s1_index:s1_index + 2]

        if len(s2) > 2:
            s2_index = random.choice(range(len(s2) - 2))
            s2 = s2[s2_index:s2_index + 2]

        return [s1, s2]