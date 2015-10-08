import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output (recommended)

print(mm)

# extract 400 LSI topics; use the default one-pass algorithm
lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=id2word, num_topics=400)

# print the most contributing words (both positively and negatively) for each of the first ten topics
lsi.print_topics(10)


# Latent Dirichlet Allocation
import logging, gensim, bz2
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# load id->word mapping (the dictionary), one of the results of step 2 above
id2word = gensim.corpora.Dictionary.load_from_text('wiki_en_wordids.txt')
# load corpus iterator
mm = gensim.corpora.MmCorpus('wiki_en_tfidf.mm')
# mm = gensim.corpora.MmCorpus(bz2.BZ2File('wiki_en_tfidf.mm.bz2')) # use this if you compressed the TFIDF output

print(mm)

# extract 100 LDA topics, using 1 pass and updating once every 1 chunk (10,000 documents)
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=1, chunksize=10000, passes=1)


# print the most contributing words for 20 randomly selected topics
lda.print_topics(20)

# extract 100 LDA topics, using 20 full passes, no online updates
lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=id2word, num_topics=100, update_every=0, passes=20)

doc_lda = lda[doc_bow]
