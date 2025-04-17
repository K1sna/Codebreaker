import json
import random
import re
import nltk
from nltk import word_tokenize, pos_tag

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')

class WordCrossover:

    @staticmethod
    def load_data(prompts, n_results):
        random_entries = random.sample(prompts, n_results)
        return random_entries

    @staticmethod
    def _extract_content_words(sentence):
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        content_word_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        content_words = [word for word, tag in tagged if tag in content_word_tags]
        return content_words

    @staticmethod
    def _word_crossover(parent1, parent2, crossover_ratio=0.5):
        words_parent1 = WordCrossover._extract_content_words(parent1)
        words_parent2 = WordCrossover._extract_content_words(parent2)

        crossover_point = int(min(len(words_parent1), len(words_parent2)) * crossover_ratio)

        # 使用 crossover 点交换实词
        child1_words = words_parent1[:crossover_point] + words_parent2[crossover_point:]
        child2_words = words_parent2[:crossover_point] + words_parent1[crossover_point:]

        # 使用原句子中的非实词填补
        child1 = WordCrossover._reconstruct_sentence(parent1, child1_words)
        child2 = WordCrossover._reconstruct_sentence(parent2, child2_words)

        return child1, child2

    @staticmethod
    def _reconstruct_sentence(original_sentence, content_words):
        tokens = word_tokenize(original_sentence)
        tagged = pos_tag(tokens)

        content_word_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        result_tokens = []
        content_word_idx = 0

        for word, tag in tagged:
            if tag in content_word_tags and content_word_idx < len(content_words):
                result_tokens.append(content_words[content_word_idx])
                content_word_idx += 1
            else:
                result_tokens.append(word)

        return ' '.join(result_tokens)

    @staticmethod
    def word_crossover_all(prompts, n_results):
        entries = WordCrossover.load_data(prompts, n_results)
        if len(entries) % 2 != 0:
            raise ValueError("The number of strings must be even.")

        pairs = [(entries[i], entries[i + 1]) for i in range(0, len(entries), 2)]

        children = []
        for pair in pairs:
            child1, child2 = WordCrossover._word_crossover(pair[0], pair[1])
            children.extend([child1, child2])

        return children

    @staticmethod
    def ran_crossover(prompts, n_results):
        random_items = random.sample(prompts, n_results)
        return random_items
