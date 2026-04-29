from datasetLoader import DatasetLoader
from classifier import Classifier
from g3pEngine import ReplacementStrategy, g3pEngineConfiguration, ClassificationStrategy
from logger import log, logStatistics, setExecutionTimer
import terminalLogic
import time
import random
import numpy as np 
import pandas as pd


from logger import log, logStatistics, setExecutionTimer, setLogFile

def launchExperiment(datasetFilePath,grammarFilePath,nFolds,maxGenerations,populationSize,
    crossProb, mutationProb, fitnessThreshold, replacementStrategy, classificationStrategy,
     positiveWeight, seed, logFilePath="log.txt", bestRulesFilePath="bestRules.txt", progress_callback=None,
     vocabStrategy="ALL", extraTerms=None, pause_event=None):
    setLogFile(logFilePath)
    random.seed(seed)
    np.random.seed(seed)
    experiment_start = time.perf_counter()

    setExecutionTimer()
    log("New Experiment")
    log("Config. details:")
    log("Dataset: " + datasetFilePath)
    log("Grammar: " + grammarFilePath)
    log("Num. Folds: " + str(nFolds))
    log("Max. Generations: " + str(maxGenerations))
    log("Population size: " + str(populationSize))  
    log(f"Fitness Threshold: {fitnessThreshold}")  
    log(f"Replacement Strategy: {replacementStrategy}")  
    log(f"Classification Strategy: {classificationStrategy}")  
    log(f"Seed: {seed}")  

    # Load dataset and extract vocabulary via text mining.
    log("Loading data...")
    step_start = time.perf_counter()
    dl = DatasetLoader()
    dataset = dl.loadfile(datasetFilePath)
    log(f"Data loaded. {time.perf_counter() - step_start:.3f}s.")

    log("Preparing text mining...")
    step_start = time.perf_counter()
    terminalLogic.VOCABULARY = dl.extractVocabulary(strategy=vocabStrategy, extra_terms=extraTerms)
    log(f"Text mining done. {time.perf_counter() - step_start:.3f}s.")
    log("Initializing classifier...")
    step_start = time.perf_counter()
    config = g3pEngineConfiguration(None, grammarFilePath, maxGenerations, populationSize, crossProb, 
                                    mutationProb, fitnessThreshold, replacementStrategy, classificationStrategy, positiveWeight)
    config.bestRulesFilePath = bestRulesFilePath
    config.datasetFilePath = datasetFilePath
    config.seed = seed
    classifier = Classifier(dataset, nFolds, config, progress_callback=progress_callback)    
    log(f"Classifier initialized. {time.perf_counter() - step_start:.3f}s.")
    log("Training...")
    step_start = time.perf_counter()
    classifier.train(pause_event=pause_event)
    log(f"Training done. {time.perf_counter() - step_start:.3f}s.")
    log("Testing...")
    step_start = time.perf_counter()
    classifier.test()
    log(f"Testing done. {time.perf_counter() - step_start:.3f}s.")
    log(f"Experiment End. {time.perf_counter() - experiment_start:.3f}s.")
    log("-------------")
    return classifier.avgMeasures, classifier.crossValidator.fitnessHistories, classifier.selectedRelevantPapers

def statistics(resultsPath='results.txt', nSeeds=1, nExperiments=1):
    if nSeeds <= 0 or nExperiments <= 0:
        raise ValueError("nSeeds and nExperiments must be greater than 0")
    if nExperiments % nSeeds != 0:
        raise ValueError("nExperiments must be divisible by nSeeds")

    df = pd.read_csv(resultsPath, sep=';')
    metrics = ["balancedAcc", "accuracy", "precision", "recall", "specificity", "time"]
    nGroups = nExperiments // nSeeds

    logStatistics(f"{nExperiments} experiments | {nSeeds} nSeeds", True)
    for label, aggregate in [("Avg.", "mean"), ("Std.", "std"), ("Max.", "max"), ("Min.", "min")]:
        logStatistics(label, True)
        for i in range(nGroups):
            currentSlice = df.iloc[(nSeeds * i):(nSeeds * (i + 1))]
            values = currentSlice[metrics].agg(aggregate).tolist()
            logStatistics(";".join(str(value) for value in values), True)


if __name__ == "__main__":
    launchExperiment(datasetFilePath = 'HallComplete.csv',grammarFilePath = 'assets/ml.xml',nFolds = 2,maxGenerations = 10,populationSize = 30,crossProb=0.9, mutationProb=0.1, fitnessThreshold = 0.4, replacementStrategy = ReplacementStrategy.NEWPOPULATION, classificationStrategy=ClassificationStrategy.CBA, positiveWeight=1.5, seed=1)
