import pandas as pd
import spacy
import itertools
import random

from happytransformer import HappyGeneration, GENSettings
from nltk.corpus import wordnet
from pyinflect import getAllInflections
from summarizer import Summarizer


class ExerciseGenerator:
    def __init__(self):
        self.__happy_gen = HappyGeneration("GPT-NEO", "EleutherAI/gpt-neo-1.3B")
        self.__file_content = ''
        self.__nlp = spacy.load('en_core_web_sm')
        self.__data_frame = None

    def get_sentences_data(self):
        return self.__data_frame

    def load_text(self, file_to_read, skip_lines_from_start=0, skip_lines_from_end=0):
        if isinstance(file_to_read, str):
            with open(file_to_read) as file:
                lines = self.__read_and_preprocess_file_content(
                    file,
                    skip_lines_from_start,
                    skip_lines_from_end)
        else:
            lines = self.__read_and_preprocess_file_content(
                file_to_read,
                skip_lines_from_start,
                skip_lines_from_end)
        self.__file_content = ' '.join(lines)

    def split_by_sentence(self):
        if self.__file_content:
            document = self.__nlp(self.__file_content)
            sentences = []
            for index, sentence in enumerate(document.sents):
                sentences.append({'id': index, 'sentence': str(sentence)})
            self.__data_frame = pd.DataFrame(sentences)
            self.__data_frame.set_index('id', inplace=True)
        return self.__data_frame

    def enrich_by_words_info(self):
        if self.__data_frame is not None:
            self.__data_frame['words'] = ''
            self.__data_frame['lemma'] = ''
            self.__data_frame['pos'] = ''
            self.__data_frame['tag'] = ''
            self.__data_frame['dep'] = ''
            for index, row in self.__data_frame.iterrows():
                current_doc = self.__nlp(row['sentence'])
                text = []
                lemma = []
                pos = []
                tag = []
                dep = []
                for i, token in enumerate(current_doc):
                    text.append(token.text)     # делим на слова
                    lemma.append(token.lemma_)  # выделяем первоначальную форму слова
                    pos.append(token.pos_)      # выделяем части речи
                    tag.append(token.tag_)      # выделяем мелкие части речи
                    dep.append(token.dep_)      # выделяем роль в предложении
                self.__data_frame.at[index, 'words'] = text
                self.__data_frame.at[index, 'lemma'] = lemma
                self.__data_frame.at[index, 'pos'] = pos
                self.__data_frame.at[index, 'tag'] = tag
                self.__data_frame.at[index, 'dep'] = dep
        return self.__data_frame

    def choose_correct_verb(self, count):
        if self.__data_frame is not None:
            required_parts_of_speech = ['PRON', 'VERB', 'ADJ']
            exercises_dataframe = self.__get_sentences_with_metadata(
                self.__data_frame,
                required_parts_of_speech,
                count)
            exercises_dataframe.reset_index(drop=True, inplace=True)
            exercises_dataframe = exercises_dataframe.apply(self.__choose_correct_verb, axis=1)
            exercises_dataframe.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)
            return exercises_dataframe

    def choose_correct_statement(self, count):
        if self.__data_frame is not None:
            required_parts_of_speech = ['NOUN', 'VERB', 'ADJ']
            training_cases = self.__get_training_cases(
                self.__data_frame,
                required_parts_of_speech,
                7)
            exercises_dataframe = self.__get_sentences_with_metadata(
                self.__data_frame,
                required_parts_of_speech,
                count)
            exercises_dataframe.reset_index(drop=True, inplace=True)
            exercises_dataframe = exercises_dataframe.apply(
                lambda row: self.__choose_correct_statement(
                    row,
                    required_parts_of_speech,
                    training_cases,
                    self.__happy_gen,
                    4),
                axis=1)
            exercises_dataframe.drop(['lemma', 'pos', 'tag', 'dep', 'words'], axis=1, inplace=True)
            return exercises_dataframe

    def choose_correct_text_summary(self, options_count=4):
        if self.__file_content is not None:
            summarizer = Summarizer()
            summarized = summarizer(self.__file_content, num_sentences=9)
            summarized_splitted = summarized.split('.')
            options = [summarized]
            args = GENSettings(do_sample=True, top_k=5, no_repeat_ngram_size=3,
                               early_stopping=True, min_length=7, max_length=200)
            while len(options) < options_count:
                last_sentence_index = random.randint(3, 6)
                beginning = '.'.join(summarized_splitted[:last_sentence_index]) + '.'
                generated = self.__happy_gen.generate_text(
                    beginning,
                    args=args)
                options.append(beginning + generated.text)

            exercises_dataframe = pd.DataFrame()
            exercises_dataframe['options'] = None
            exercises_dataframe['options'] = exercises_dataframe['options'].astype('object')
            exercises_dataframe.at[0, 'options'] = options
            exercises_dataframe['sentence'] = ''
            exercises_dataframe['exercise_context'] = ''
            exercises_dataframe['answer'] = summarized
            exercises_dataframe['description'] = 'Выберите утверждение, ' \
                                                 'которое ближе всего соотносится с текстом'
            exercises_dataframe['exercise_type'] = 'select_correct_summary'
            return exercises_dataframe

    @staticmethod
    def __read_and_preprocess_file_content(file, skip_lines_from_start, skip_lines_from_end):
        lines = [line.rstrip() for line in file]
        lines = list(filter(lambda x: x != '', lines))
        lines = lines[skip_lines_from_start:len(lines) - skip_lines_from_end]
        return lines

    @staticmethod
    def __get_sentence_from_keywords(keywords, training_cases, happy_gen):
        prompt = ExerciseGenerator.__create_prompt(keywords, training_cases)
        args_top_k = GENSettings(do_sample=True, top_k=5, no_repeat_ngram_size=3,
                                 early_stopping=True, min_length=7, max_length=50)
        result = happy_gen.generate_text(prompt, args=args_top_k)
        return result.text

    @staticmethod
    def __get_training_cases(df, parts_of_speech, count):
        df_slice = ExerciseGenerator.__get_sentences_with_metadata(
            df,
            parts_of_speech,
            count,
            lambda df_row: '"' not in df_row['sentence'])
        training_cases = ''
        for row_index, row in df_slice.iterrows():
            keywords = 'Keywords: '
            output = 'Output: ' + row['sentence'] + '\n'
            cases_separator = '###\n'
            words = ExerciseGenerator.__get_words_by_parts_of_speach(row, parts_of_speech)
            keywords += ', '.join(words) + '\n'
            training_cases += keywords + output + cases_separator
        training_cases = training_cases.rstrip('\n')
        return training_cases

    @staticmethod
    def __get_words_by_parts_of_speach(row, parts_of_speech):
        indices = []
        for element in parts_of_speech:
            indices += [i for i, x in enumerate(row['pos']) if x == element]
        indices.sort()
        words = []
        for index in indices:
            words.append(row['words'][index])
        return words

    @staticmethod
    def __create_prompt(keywords, training_cases):
        keywords_string = ', '.join(keywords)
        prompt = training_cases + '\nKeywords: ' + keywords_string + '\nOutput:'
        return prompt

    @staticmethod
    def __get_sentences_with_metadata(df, parts_of_speech, count, additional_conditions=None):
        if additional_conditions is not None:
            return df.loc[df.apply(
                lambda p: all(x in p['pos'] for x in parts_of_speech)
                          and len(p['words']) > 7
                          and additional_conditions(p), axis=1)]\
                .sample(count)
        else:
            return df.loc[df.apply(
                lambda p: all(x in p['pos'] for x in parts_of_speech)
                          and len(p['words']) > 7, axis=1)]\
                .sample(count)

    @staticmethod
    def __choose_correct_verb(row):
        verb_index = row['pos'].index('VERB')
        verb = row['words'][verb_index]
        lemma = row['lemma'][verb_index]
        sentence_without_word = ' '.join(
            row['words'][:verb_index] + ['...'] + row['words'][verb_index + 1:])

        # Начинаем собирать варианты ответов для упражнения. Первым добавляем правильный ответ
        options = [verb]

        # Получаем все формы слова по лемме
        inflections = getAllInflections(lemma)

        # Берем из всех форм только глагольные
        for key in inflections:
            if key.startswith('V'):
                options.append(inflections[key][0])

        # Убираем повторения через преобразование к set и обратно преобразуем к list
        options = list(set(options))

        row['exercise_context'] = sentence_without_word
        row['answer'] = verb
        row['options'] = options
        row['description'] = 'Выберите правильный глагол'
        row['exercise_type'] = 'select_correct_verb'
        return row

    @staticmethod
    def __choose_correct_statement(row, parts_of_speech, training_cases, happy_gen, required_options_count):
        words = ExerciseGenerator.__get_words_by_parts_of_speach(row, parts_of_speech)
        adjectives = ExerciseGenerator.__get_words_by_parts_of_speach(row, ['ADJ'])
        antonyms = {}
        for adj in adjectives:
            for syn in wordnet.synsets(adj):
                for lm in syn.lemmas():
                    if lm.antonyms():
                        if adj not in antonyms:
                            antonyms[adj] = []
                        antonyms[adj].append(lm.antonyms()[0].name())

        options = [row['sentence']]
        for adj in adjectives:
            if len(options) >= required_options_count:
                break
            if adj in antonyms:
                for ant in antonyms[adj]:
                    new_keywords = list(map(lambda x: x.replace(adj, ant), words))
                    options.append(ExerciseGenerator.__get_sentence_from_keywords(
                        new_keywords,
                        training_cases,
                        happy_gen))
                    if len(options) >= required_options_count:
                        break

        while len(options) < required_options_count:
            options.append(ExerciseGenerator.__get_sentence_from_keywords(
                words,
                training_cases,
                happy_gen))

        options = list(map(lambda x: x.split('.')[0].strip() + '.', options))
        permutations = list(itertools.permutations(options))
        options = permutations[random.randint(0, len(permutations))]

        row['exercise_context'] = ''
        row['answer'] = row['sentence']
        row['options'] = options
        row['description'] = 'Выберите правильное утверждение'
        row['exercise_type'] = 'select_correct_statement'
        return row
