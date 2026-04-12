from candidatePaper import CandidatePaper
import os
import pandas as pd
import csv
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

"""This class is prepared to load, save files and to perform text mining
following Zu&Menzies approach.
Reference: https://github.com/fastread/src/blob/master/src/util/mar.py"""
class DatasetLoader():
    def __init__(self):
        self.fea_num = 10        
        self.filename=""
        self.name=self.filename.split(".")[0]
        self.body={}
        self.papers=[]
        self.voc=[]

    def loadCSV(filepath):
        papers = []
        frame = pd.read_csv(filepath)
        #print(frame)
        for index, row in frame.iterrows():
            paper = CandidatePaper(row[0],row[1],row[2],row[3],row[4])
            papers.append(paper)
        return papers

    def loadfile(self, filename):
        self.filename = filename
        # Check if the file is an absolute path and exists, or relative to the root
        if os.path.isabs(filename) and os.path.exists(filename):
            full_path = filename
        elif os.path.exists(filename):
            full_path = filename
        else:
            full_path = "datasets/" + str(self.filename)
            
        with open(full_path, "r", encoding="UTF-8-sig") as csvfile:
            headerLine = csvfile.readline()
            # Determine delimiter based on the first line
            delimiter = ',' if ',' in headerLine and (';' not in headerLine or headerLine.count(',') >= headerLine.count(';')) else ';'
            csvfile.seek(0)
            content = [x for x in csv.reader(csvfile, delimiter=delimiter)]
        header = [h.strip() for h in content[0]]
        header_lower = [h.lower() for h in header]

        # Check if enrichment is needed (missing Title or Abstract but has DOI)
        title_exists = any(h in ["document title", "title"] for h in header_lower)
        abstract_exists = any(h in ["abstract", "description"] for h in header_lower)
        doi_exists = any(h in ["doi"] for h in header_lower)
        
        if (not title_exists or not abstract_exists) and doi_exists:
            from datasetCrawler import enrich_dataset
            print(f"Dataset {full_path} is missing critical info. Triggering DOI enrichment...")
            new_path = enrich_dataset(full_path)
            if new_path != full_path:
                return self.loadfile(new_path) # Reload with the enriched file

        def get_col_data(possible_names, default_val=None):
            for name in possible_names:
                try:
                    idx = header_lower.index(name.lower())
                    return [c[idx] if idx < len(c) else default_val for c in content[1:]]
                except ValueError:
                    continue
            return [default_val] * (len(content) - 1)

        self.body["Document Title"] = get_col_data(["Document Title", "Title"])
        self.body["Abstract"] = get_col_data(["Abstract", "Description"])
        self.body["Year"] = get_col_data(["Year", "Date", "Cover Date"])
        self.body["PDF Link"] = get_col_data(["PDF Link", "Link", "url"], "undetermined")
        self.body["doi"] = get_col_data(["doi", "DOI"], "undetermined")
        self.body["label"] = get_col_data(["label", "isCandidate", "Candidate"], "unknown")
        self.body["nCites"] = get_col_data(["nCites", "citedby_count", "citations", "Cites"], 0)
        self.body["authorCount"] = get_col_data(["authorCount", "author_count", "nAuthors", "authors"], 0)
        self.body["aggregationType"] = get_col_data(["aggregationType", "type", "Aggregation Type"], "undetermined")
        self.body["time"] = get_col_data(["time"], 0)

        self.papers = []
        
        for index in range(len(self.body["Document Title"])) :
            self.papers.append(CandidatePaper(self.body["Document Title"][index], self.body["Abstract"][index],
            self.body["Year"][index],self.body["PDF Link"][index], self.body["label"][index],
            self.body["doi"][index],self.body["nCites"][index],self.body["authorCount"][index],
            self.body["aggregationType"][index]))

        return self.papers
    
    
    def extractVocabulary(self, strategy="ALL", extra_terms=None):
        if strategy == "POSITIVE":
            content = [paper.documentTitle + " " + paper.abstract for paper in
                      self.papers if paper.getIsCandidate()]
        elif strategy == "NEGATIVE":
            content = [paper.documentTitle + " " + paper.abstract for paper in
                      self.papers if not paper.getIsCandidate()]
        else: # Default: ALL
            content = [paper.documentTitle + " " + paper.abstract for paper in self.papers]

        self.voc = self.getRelevantWords(content)
        
        # Add manually specified terms
        if extra_terms:
            if isinstance(extra_terms, str):
                terms = [t.strip() for t in extra_terms.split(",") if t.strip()]
            else:
                terms = extra_terms
            
            # Use stemmer for manual terms to maintain consistency
            from nltk.stem import PorterStemmer
            porter = PorterStemmer()
            stemmed_terms = [porter.stem(t.lower()) for t in terms]
            
            for term in stemmed_terms:
                if term not in self.voc:
                    self.voc.append(term)

        # Apply vocabulary to terminalLogic
        import terminalLogic
        terminalLogic.VOCABULARY = self.voc
        
        return self.voc
    
    def getRelevantWords(self, content):
        ### Feature selection by tfidf in order to keep vocabulary ###
        tfidfer = TfidfVectorizer(lowercase=True, stop_words="english", norm=None, use_idf=True, smooth_idf=False,
                                sublinear_tf=False,decode_error="ignore",)
        tfidf = tfidfer.fit_transform(content)
        weight = tfidf.sum(axis=0).tolist()[0]
        kept = np.argsort(weight)[-self.fea_num:]
        # Define the stemmer
        porter = nltk.stem.PorterStemmer()
        voc = np.array(list(tfidfer.vocabulary_.keys()))[np.argsort(list(tfidfer.vocabulary_.values()))][kept]
        tokens_stemmed = [w.replace(w, porter.stem(w)) for w in voc]
        voc = list(dict.fromkeys(tokens_stemmed))
        return voc