"""Represents an instance of the dataset"""
class CandidatePaper:
    __slots__ = ("documentTitle", "abstract", "year", "pdfLink","doi","nCites","nAuthors","aggregationType","isCandidate")

    def __init__(self, documentTitle, abstract, year, pdfLink, isCandidate, doi="",nCites=0,nAuthors=0,aggregationType=""):
        self.documentTitle = documentTitle
        self.abstract = abstract
        self.year = year
        self.pdfLink = pdfLink
        self.doi = doi
        self.nCites = nCites
        self.nAuthors = nAuthors
        self.aggregationType = aggregationType
        self.isCandidate = isCandidate

    """Compares with 'yes' the value of the isCandidate column of the dataset"""
    def getIsCandidate(self):
        val = str(self.isCandidate).lower().strip()
        return val in ['yes', '1', 'true', 'y']
    
    def __str__(self):
        return str(self.doi) + ' -> ' + str(self.isCandidate)

 
        

