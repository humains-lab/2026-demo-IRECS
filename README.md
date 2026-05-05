# 2026-demo-IRECS
Functional demonstration of the automatic selection of candidate studies in systematic literature reviews using evolutionary computation.

# Automatic selection of primary studies in systematic reviews with evolutionary rule-based classification
_Supplementary material_ (September, 2025)

## Authors
- José de la Torre-López
- Aurora Ramírez 
- José Raúl Romero (corresponding author)

Dept. Computer Science and Artificial Intelligence, University of Córdoba, 14071, Córdoba, Spain


## Abstract
Searching, filtering and analysing scientific literature are time-consuming tasks when performing a systematic literature review. With the rise of artificial intelligence, some steps in the review process are progressively being automated. In particular, machine learning for automatic paper selection can greatly reduce the effort required to identify relevant literature in scientific databases. We propose an evolutionary machine learning approach, called IRECS, to automatically determine whether a paper retrieved from a literature search process is relevant. IRECS builds an interpretable rule-based classifier using grammar-guided genetic programming. The use of a grammar to define the syntax and the structure of the rules allows IRECS to easily combine the usual textual information with other bibliometric data not considered by state-of-the-art methods. Our experiments demonstrate that it is possible to generate accurate classifiers without impairing interpretability and using configurable information sources not supported so far.

## Supplementary material

### Datasets

We use five datasets to conduct our experiments. Some datasets were provided by previous studies, although we have extended them to include bibliometric information. Also, we have created a new dataset. All datasets used in the experiments are the [datasets](https://github.com/jrromero/irecs/tree/main/datasets) folder. The original datasets are:

- We use the datasets Hall, Wahono and Kitchenham provided by Z. Yu and T. Menzies in their paper [FAST2: An intelligent assistant for finding relevant papers](https://doi.org/10.1016/j.eswa.2018.11.021). The dataset can be downloaded from the replication package in [Github](https://github.com/fastread/src). 
- The Muthu dataset is part of the collection provided by the ASReview tool, available on [Github](https://github.com/asreview/synergy-dataset).


### Research questions

Our research aim to answer the following research questions:

- RQ1: Which parameter configuration offers the best performance for automatic classification of relevant studies?
- RQ2: Do bibliometric operators contribute to improved classification performance?
- RQ3: What are the characteristics of the rules selected to build rule-based classifiers for automatic paper selection?
- RQ4: How does IRECS perform compared to a state-of-the-art black box classifier?

### Content

This repository includes the following directories:

- [code](https://github.com/jrromero/irecs/tree/main/code): Source code of IRECS, developed in Python. A requirements.txt file is included to install required packages. Assets are compressed as ZIP files due to the size limit.
- [data](https://github.com/jrromero/irecs/tree/main/datasets): The datasets used for experimentation (compressed as ZIP files due to the size limit).
- [results](https://github.com/jrromero/irecs/tree/main/results): Detailed results of classification performance. The folder includes a spreadsheet with the experimental results by RQ, files with the best rules found by IRECS for each dataset (RQ3) and statistical tests (RQ4).
- [uploads]: Folder used to cache uploaded datasets in the web tool.
- [web-tool] Source code for UI web-based tool designed to connect with IRECS core. It uses Next.js as front-end and FastAPI in Python as back-end. Please, check DEPLOYMENT.md(https://github.com/humains-lab/2026-demo-IRECS/blob/main/DEPLOYMENT.md) for further details.
