# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel

from corpus import *
from settings import *


class LMDL_LDA():

	def __init__(self):
		self.lmdl = LMDL_Corpus()
		self.texts = self.lmdl.get_corpus_texts_words()
		self.dictionary = Dictionary(self.texts)
		self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
		self.lda = LdaModel(self.corpus, num_topics=LDA_NUM_TOPICS, id2word=self.dictionary)

	def print_topics(self):
		return self.lda.print_topics(LDA_NUM_TOPICS)

	def get_document_topics(self, document_name):
		document_tokens = self.lmdl.token_list_processed(document_name)
		topics = self.lda.get_document_topics(self.dictionary.doc2bow(document_tokens), minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
		show_topics_list = []
		for topic in topics:
			lda_topic = self.lda.show_topic(topic[0], topn=10)
			show_topics_list.append(lda_topic)
		return show_topics_list

	def top_topics(self):
		return self.lda.top_topics(corpus=self.corpus, texts=self.texts, dictionary=self.dictionary, window_size=None, coherence='u_mass', topn=20, processes=-1)

if __name__ == '__main__':
	lda = LMDL_LDA()
	#print(lda.print_topics())
	#print(lda.get_document_topics('acurrucadas.html'))
	print(lda.top_topics())


