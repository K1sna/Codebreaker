import json
import random
import string

class TextMutator:
    def __init__(self, corpus_file):
        self.corpus_file = corpus_file

    def load_data(self, texts):
        return texts
    
    def oneinsert(self, sentence, word_or_phrase):
        words = sentence.split()
        insert_position = random.randint(0, len(words))
        words.insert(insert_position, word_or_phrase)
        new_sentence = ' '.join(words)
        new_sentence = new_sentence.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')

        return new_sentence

    def mutate_text(self, text, mutation_rate=0.4):
        mutated_text = ""
        for char in text:
            if random.random() < mutation_rate:
                mutation_type = random.choices(
                    ["insert", "delete", "replace", "rearrange"],
                    weights=[0.3, 0.1, 0.4, 0.2],
                    k=1
                )[0]
                if mutation_type == "insert":
                    mutated_text += char + random.choice(string.ascii_letters)
                elif mutation_type == "delete":
                    continue
                elif mutation_type == "replace":
                    mutated_text += random.choice(string.ascii_letters)
                elif mutation_type == "rearrange":
                    if len(mutated_text) > 0:
                        mutated_text = mutated_text[:-1] + char + mutated_text[-1]
            else:
                mutated_text += char
        return mutated_text

    def mutate_with_corpus(self, text, mutation_rate=0.1):
        with open(self.corpus_file, 'r', encoding='utf-8') as file:
            corpus = [line.strip() for line in file]

        mutated_text = ""
        words = text.split()

        for word in words:
            if random.random() < mutation_rate:
                mutated_text += random.choice(corpus) + " "
            else:
                mutated_text += word + " "

        return mutated_text.strip()

    def mutate_text_list(self, texts):
        mutated_texts = []
        for text in texts:
            mutated_text = self.mutate_with_corpus(text)
            mutated_text = self.mutate_text(mutated_text)
            mutated_texts.append(mutated_text)
        return mutated_texts
