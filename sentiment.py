#ag!/usr/bin/python
 
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import sys
from nltk.corpus import movie_reviews
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
 
def subj(subjLine, subj1):
    subjgen = subjLine.lower()  
    if subjgen.find(subj1) != -1:
        subject = subj1
        return subject
    else:
        subject = "No match"
        return subject
 
 
def main(argv):
    negative = movie_reviews.fileids('neg')
    positive = movie_reviews.fileids('pos')
 
    negative_feats = [(word_feats(movie_reviews.words(fileids=[f])), 'negative') for f in negative]
    positive_feats = [(word_feats(movie_reviews.words(fileids=[f])), 'positive') for f in positive]
 
    total_feats =  positive_feats+negative_feats
    classifier = NaiveBayesClassifier.train(total_feats)
 
    categories = ["news", "sports", "fashion", "shopping", "finance", "politics"]
    for line in sys.stdin:
        try:
            tolk_posset = word_tokenize(line.rstrip())
            d = word_feats(tolk_posset)
            for category in categories:
                subjectFull = subj(line, category)
                if not subjectFull == "No match":
                    print "LongValueSum: " +subjectFull + "," + classifier.classify(d) + "\t" + "1"                    
        except:
                continue     
 
 
if __name__ == "__main__":
    main(sys.argv)