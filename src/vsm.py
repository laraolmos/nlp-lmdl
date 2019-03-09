# -*- coding: utf-8 -*-
__author__ = 'Lara Olmos Camarena'


from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from gensim.models import TfidfModel

from numpy import log1p

from corpus import LMDL_Corpus
from export import export_dictionary_to_txt
from export import export_to_txt


class LMDL_VSM():

	def __init__(self):
		self.corpus = LMDL_Corpus()
		self.mapping = Dictionary(self.corpus.get_processed_documents())
		self.bow_list = self._corpus_bow()
		self.inverted_index = self._inverted_index_matrix()
		self.tfidf_model = TfidfModel(self.bow_list, dictionary=self.mapping, id2word=self.mapping.token2id)
		self.log_inverted_index = log1p(self.inverted_index.tocsc())

	def _corpus_bow(self):
		sorted_filename = self.corpus.get_filenames()
		ordered_bow = []
		for filename in sorted_filename:
			dt = self.corpus.document_terms(filename)
			ordered_bow.append(self.mapping.doc2bow(dt))
		return ordered_bow

	def _inverted_index_matrix(self):
		# term-document sparse matrix with TF weight
		sparse_matrix = corpus2csc(corpus=self.bow_list, num_terms=self.corpus.vocabulary_size(), 
			num_docs=self.corpus.number_of_documents())
		return sparse_matrix.tocsr()

	def verbose_inverted_sparse_index(self):
		inverted_index = {}
		filenames = self.corpus.get_filenames()
		for term in self.corpus.get_vocabulary():
			if term in self.mapping.token2id.keys():
				token_id = self.mapping.token2id[term]
				token_row = self.inverted_index.getrow(token_id).toarray().tolist()
				inverted_index[term] = list(zip(filenames, token_row[0]))
		return inverted_index

	def verbose_inverted_index(self):
		inverted_index = {}
		filenames = self.corpus.get_filenames()
		for term in self.corpus.get_vocabulary():
			if term in self.mapping.token2id.keys():
				token_id = self.mapping.token2id[term]
				token_row = self.inverted_index.getrow(token_id).toarray().tolist()
				sparse_zip = list(zip(filenames, token_row[0]))
				no_zero = [(ubication, weight) for (ubication, weight) in sparse_zip if weight > 0]
				inverted_index[term] = no_zero
		return inverted_index

	def documents_tfidf(self):
		filenames = self.corpus.get_filenames()
		for file in filenames:
			token_col = self.inverted_index.getcol(filenames.index(file)).toarray().tolist()
			sparse_bow = list(zip(list(range(0, self.corpus.vocabulary_size())), token_col))
			sparse_bow = [(ubication, weight) for (ubication, [weight]) in sparse_bow]
			vector_list = self.tfidf_model.__getitem__(sparse_bow, eps=-1)
			vector_str = [str(weight) for (ubication, weight) in vector_list]
			export_to_txt('tfidf\\' + file.replace('html','txt'), ' '.join(vector_str))

	def documents_log1p(self):
		filenames = self.corpus.get_filenames()
		for file in filenames:
			vector_list = self.log_inverted_index.getcol(filenames.index(file)).toarray().tolist()
			vector_str = [str(element) for [element] in vector_list]
			export_to_txt('locallog\\' + file.replace('html','txt'), ' '.join(vector_str))


if __name__ == '__main__':

	lmdl_vsm = LMDL_VSM()
	export_to_txt('vocabulary.txt', ' '.join(lmdl_vsm.corpus.vocabulary))
	export_dictionary_to_txt('inverted_sparse_file.txt', lmdl_vsm.verbose_inverted_sparse_index())
	export_dictionary_to_txt('inverted_file.txt', lmdl_vsm.verbose_inverted_index())

	lmdl_vsm.documents_tfidf()
	lmdl_vsm.documents_log1p()
