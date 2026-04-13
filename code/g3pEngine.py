import copy
from functools import reduce
from itertools import count
import random
import numpy as np 
from typing import List
from evaluator import Evaluator
from g3p import SyntaxTree, SyntaxTreeSchema, cxSubtree, mutSubtree
from enum import Enum
from grammarParser import parse
from logger import log

class ReplacementStrategy(Enum):
    PUREGENERATIONAL = 1
    ELITIST = 2
    NEWPOPULATION = 3

class ClassificationStrategy(Enum):
    CBA = 1
    CMAR = 2
    MULTISTRATEGY = 3
    CPAR = 4
    SCBA = 5

class g3pIndividual:
    def __init__(self, syntaxTree: SyntaxTree):
        self.syntaxTree = syntaxTree
        self.support = 0.0
        self.confidence = 0.0
        self.lift = 0.0
        self.suppAnt = 0.0
        self.suppCon = 0.0
        self.fitness = 0.0
        self.laplace = 0.0
        self.score = 0.0
        self.antcCovered = 0
        self.consqCovered = 0
        self.instancesCovered = 0
        self.normalizeRule()
        

    def normalizeRule(self):
        nElems = len(self.syntaxTree)
        if self.syntaxTree[nElems - 3].symbol == "notEquals":
            self.syntaxTree[nElems - 3].symbol = "equals"
            self.syntaxTree[nElems - 3].code = "cmpEquals"
            self.syntaxTree[nElems - 1].symbol =  False if self.syntaxTree[nElems - 1].symbol else True

    def reverseRule(self):
        clone = copy.deepcopy(self)
        nElems = len(clone.syntaxTree)
        clone.syntaxTree[nElems - 1].symbol =  False if clone.syntaxTree[nElems - 1].symbol else True
        clone.support = 0
        clone.fitness = 0
        clone.instancesCovered = 0
        clone.antcCovered = 0
        clone.consqCovered = 0
        clone.instancesCovered = 0
        clone.confidence = 0
        clone.suppAnt = 0
        clone.suppCon = 0
        clone.lift = 0
        clone.laplace = 0.0
        clone.score = 0.0
        return clone

    def setSupport(self, support):
        self.support = support

    def setConfidence(self, confidence):   
        self.confidence = confidence
    
    def getSyntaxTree(self):
        return self.syntaxTree
    
    def getFitness(self):
        return self.fitness

    def getRuleSize(self):
        return len(self.syntaxTree)

    def isPositiveCandidate(self):
        """Determines whether the rule identifies positive candidates or not"""
        genotype = self.syntaxTree.getTerminals()
        return (
            (str(genotype[len(genotype)-3].symbol) == 'equals' and str(genotype[len(genotype)-1].symbol) == 'True') 
        or 
            (str(genotype[len(genotype)-3].symbol) == 'notEquals' and str(genotype[len(genotype)-1].symbol) == 'False'))
        
    
    def __str__(self):
        s = ""
        for terminal in self.syntaxTree.getTerminals():
            s += str(terminal)
            s += " "
        return s.strip()
    
    def __eq__(self, other):
        if not isinstance(other, g3pIndividual):
            return False
        # Compare based on the full syntax tree symbols sequence
        if len(self.syntaxTree) != len(other.syntaxTree):
            return False
        for s, o in zip(self.syntaxTree, other.syntaxTree):
            # Handle numpy arrays specifically for equality comparison
            if isinstance(s.symbol, np.ndarray) or isinstance(o.symbol, np.ndarray):
                if not np.array_equal(s.symbol, o.symbol):
                    return False
            elif s.symbol != o.symbol:
                return False
        return True

    def __hash__(self):
        # Use a tuple of symbols for hashing. Convert unhashable ndarrays to tuples.
        symbols = []
        for s in self.syntaxTree:
            if isinstance(s.symbol, np.ndarray):
                symbols.append(tuple(s.symbol))
            else:
                symbols.append(s.symbol)
        return hash(tuple(symbols))

    def __lt__(self, other):
        # Deterministic sorting for reproducibility when hashes/identities are tied or for list(set())
        # We use the string representation as a stable tie-breaker
        return str(self) < str(other)

class g3pEngineConfiguration:
    def __init__(self, dataset, grammarFilePath,maxGenerations = 100, populationSize = 1000, crossProb = 0.9, mutationProb = 0.1,
                fitnessThreshold = 0.6,
                  replacementStrategy = ReplacementStrategy.PUREGENERATIONAL, classificationStrategy = ClassificationStrategy.CBA,
                  positiveWeight = 1.5):
       self.populationSize = populationSize
       self.maxGenerations = maxGenerations
       self.crossProb = crossProb
       self.mutationProb = mutationProb
       self.grammarFilePath = grammarFilePath
       self.replacementStrategy = replacementStrategy
       self.bestRulesFilePath = "bestRules.txt"
       self.storeBestRulesIntraGeneration = True
       self.supportThreshold = 0.5
       self.confidenceThreshold = 0.6
       self.fitnessThreshold = fitnessThreshold
       self.dataset = dataset
       self.classificationStrategy = classificationStrategy
       self.positiveWeight = positiveWeight
       self.datasetFilePath = ""
       self.seed = 0

    def getMaxGenerations(self):
        return self.maxGenerations
    
    def getPopulationSize(self):
        return self.populationSize
    
    def getCrossProbability(self):
        return self.crossProb
    
    def getMutationProbability(self):
        return self.mutationProb
    
    def getReplacementStrategy(self):
        return self.replacementStrategy

class g3pEngine:
    """This class represents a G3P engine that works closely with G3P module.

    It requires a grammar (SyntaxTreeSchema), a dataset to evaluate each 
    individual created (currently, a list of CandidatePapers) and a config object
    for the engine with population size, max. generations, replacement strategy
    and probabilities although default values are given."""
    def __init__(self, config: g3pEngineConfiguration, progress_callback=None, totalFolds=1, foldIdx=0):
        self.config = config
        self.progress_callback = progress_callback
        self.totalFolds = totalFolds
        self.foldIdx = foldIdx
        self.populationSize = config.getPopulationSize()
        self.maxGenerations = config.getMaxGenerations()
        self.crossProb = config.getCrossProbability()
        self.mutationProb = config.getMutationProbability()
        self.replacementStrategy = config.getReplacementStrategy()
        self.individuals:List[g3pIndividual] = []
        self.bestOverallIndividuals:List[g3pIndividual] = []
        self.currentGen = 0
        self.parentsSelected = []
        self.childrenCurrentGen = []
        self.maxGrammarDerivations = 15;    
        self.dataset = config.dataset 
        self.pruneThreshold = 1  
        self.avgFitnessConverge = False 
        self.avgFitnessThreshold = 0.9
        self.minimumNegativeIndividuals = 5
        self.positiveWeight = config.positiveWeight
        self.fitnessHistory = [] # To capture fitness over generations
        
        #Parse grammar
        (rootSymbol, terminals, nonTerminals) = parse(config.grammarFilePath)
        #Create SyntaxTreeSchema up to maxGrammarDerivations derivations
        self.grammar = SyntaxTreeSchema(self.maxGrammarDerivations, rootSymbol, terminals, nonTerminals)


    def cover(instance, individual):
        antcResult, consqResult, ruleResult = Evaluator(individual).evaluatePaper(instance)
        if antcResult:
            individual.antcCovered += 1
        if consqResult:
            individual.consqCovered += 1
        if ruleResult:
            individual.instancesCovered += 1
        return antcResult, consqResult, ruleResult  
    
    def coverAll(self, individual):
        [g3pEngine.cover(instance, individual) for instance in self.dataset]
        return individual
    
    def generateIndividuals(self):
        """Creates the set of initial individuals"""
        for i in range(self.populationSize):
            self.individuals.append(g3pIndividual(self.grammar.createSyntaxTree()))

    def start(self, pause_event=None):
        """Launches the whole genetic engine with the established configuration.
        Stops when maxGeneration is reached."""
        self.generateIndividuals()
        #[print(str(i)) for i in self.individuals]
        #self.computeGenerationFitness()
        if self.config.storeBestRulesIntraGeneration:
            file = open(self.config.bestRulesFilePath, 'a')
            file.write('------\nNew Fold Rules\n')
            file.flush()
            file.close()

        #print('Generation... ',end=' - ')
        while self.currentGen < self.maxGenerations: # and not self.avgFitnessConverge:
            # Check for pause
            if pause_event and pause_event.is_set():
                import time
                while pause_event.is_set():
                    time.sleep(0.5)
            #print(str(self.currentGen), end=',')
            #initial selection and crossover
            self.selectionPhase()

            #selection for the new generation
            self.crossPhase()

            #mutation
            self.mutateGeneration()

            self.computeGenerationFitness()

            #replacement
            self.replacementPhase()
            
            
            #######
            self.removeClones()
            #######

            #print("La generación queda así...")
            
            self.currentGen += 1
            if self.currentGen % 2 == 0:
                log(f'Current gen.: {self.currentGen} & AVG fitness: {self.averageFitness}')
            
            if self.progress_callback:
                # Calculate overall progress: (folds done + current fold progress) / total folds
                fold_progress = self.currentGen / self.maxGenerations
                overall_progress = (self.foldIdx + fold_progress) / self.totalFolds
                self.progress_callback(overall_progress * 100)

    def sortIndividuals(self, individuals):
        """Sorts individuals by fitness descending."""
        individuals.sort(key=g3pIndividual.getFitness,reverse=True) 

    def computeGenerationFitness(self):
        """Evaluate fitness for the whole current generation and sort the current generation."""
        self.individuals = list(map(self.fitness,self.individuals))

        #First individuals are the best in this generation
        self.sortIndividuals(self.individuals)
        #[print(str(i),"Fitness: ",i.getFitness(),"Confidence: ",i.confidence,sep=" ") for i in self.individuals[:10]]
        #print("---")

        #if the average fitness is above a threshold stop the engine, we are converging.
        fitnesses = []
        [fitnesses.append(i.getFitness()) for i in self.bestOverallIndividuals]
        self.averageFitness = 0
        if len(fitnesses) != 0:
            self.averageFitness = reduce(lambda a, b: a + b, fitnesses) / len(fitnesses)
            self.avgFitnessConverge = self.averageFitness >= self.avgFitnessThreshold 
            self.fitnessHistory.append(self.averageFitness)

    def fitness(self, individual: g3pIndividual):
        """Evaluates a g3pIndividual and returns the individual with its support and confidence."""
        positiveInstances = [i for i in self.dataset if i.getIsCandidate()]
        negativeInstances = [i for i in self.dataset if not i.getIsCandidate()]
        evaluatorAll = Evaluator(individual, self.dataset)

        #If it is highly unbalanced
        if len(self.dataset) > 0 and len(positiveInstances) > 0 and len(positiveInstances)/len(self.dataset) < 0.2:
            fitness = 0.0
            weightPositive = self.positiveWeight 
            individual = evaluatorAll.evaluate()  
            if individual.isPositiveCandidate():
                evaluatorPartial = Evaluator(individual, negativeInstances)
                evaluatorPartial.evaluate()
                fitness = (evaluatorAll.satisfiedInst/len(positiveInstances))-(evaluatorPartial.satisfiedAntc/len(negativeInstances))*weightPositive
            else:
                evaluatorPartial = Evaluator(individual, positiveInstances)
                evaluatorPartial.evaluate()
                fitness = (evaluatorAll.satisfiedInst/len(negativeInstances))-(evaluatorPartial.satisfiedAntc/len(positiveInstances))  
            individual = evaluatorAll.evaluate()  
            individual.fitness = fitness
            return individual
        else:
            individual = evaluatorAll.evaluate()  
            individual.fitness = individual.support
            return individual

    def selectionPhase(self):
        """
        Selects randomly N (population size) to be parents for the new generation.
        Tournament selection is implemented for pair comparison.
        """
        #select randomly N (population size) individuals of current generation
        for i in range(0, self.populationSize):
            index = np.random.randint(0,self.populationSize)
            ind1 = self.individuals[index]
            # while self.individuals[index] in self.parentsSelected:
            #     index = random.randrange(0,self.populationSize-1)
            #     ind1 = self.individuals[index]
            index2 = np.random.randint(0,self.populationSize)
            while index == index2: #make sure parents are different
                index2 = np.random.randint(0,self.populationSize)
            ind2 = self.individuals[index2]
            self.parentsSelected.append(ind1 if (ind1.getFitness() >= ind2.getFitness()) else ind2 ) #Tournament selection

    def crossPhase(self):
        """Executes cross phase for the parents selected in the selection phase"""
        for i in range(0, len(self.parentsSelected)-1, 2):
            if (np.random.uniform(0, 1) < self.crossProb) and str(self.parentsSelected[i]) != str(self.parentsSelected[i+1]): #Don't cross twins
                #print("***Cruzando ",str(self.parentsSelected[i]),str(self.parentsSelected[i+1]),sep="\n")
                child1, child2 = cxSubtree(self.parentsSelected[i].getSyntaxTree(), self.parentsSelected[i+1].getSyntaxTree())
                self.childrenCurrentGen.append(g3pIndividual(child1))
                self.childrenCurrentGen.append(g3pIndividual(child2))
                #print("***Nuevos individuos ",str(child1),str(child2),sep="\n")
            else:
                self.childrenCurrentGen.append(self.parentsSelected[i])
                self.childrenCurrentGen.append(self.parentsSelected[i+1])


    def mutate(self, ind):
        """Mutates a given ind depending on mutationProb of the engine configuration"""
        if (np.random.uniform(0, 1) < self.mutationProb) and ind.getFitness() < 0.5:
                #print("***Mutando ",str(ind),sep="\n")
                ind.syntaxTree = mutSubtree(ind.syntaxTree, self.grammar)
                #print("***Nuevo individuo:" , str(ind),sep="\n")
        ind.normalizeRule()
        return ind

    def mutateGeneration(self):
        """Mutates the whole current generation of children"""
        self.childrenCurrentGen = list(map(self.mutate , self.childrenCurrentGen))
    
    def replacementPhase(self):
        """Executes the replacement strategy established in the engine configuration"""
        # if self.config.storeBestRulesIntraGeneration:
        #     [self.storeRule(individual) for individual in self.individuals if (individual.support >= self.config.supportThreshold 
        #                                                                     and individual.confidence >= self.config.confidenceThreshold)]
            
        if self.config.storeBestRulesIntraGeneration:
            [self.storeRule(individual) for individual in self.individuals if (individual.fitness > self.config.fitnessThreshold)]
        
        if self.replacementStrategy == ReplacementStrategy.ELITIST:
            self.parentsSelected.extend(self.childrenCurrentGen)
            self.sortIndividuals(self.parentsSelected)
            self.individuals = self.parentsSelected[0:self.populationSize]
        if self.replacementStrategy == ReplacementStrategy.PUREGENERATIONAL:
            self.individuals = self.childrenCurrentGen
        if self.replacementStrategy == ReplacementStrategy.NEWPOPULATION:
            self.individuals = []
            self.generateIndividuals()

        # Generate masive POSITIVES
        if self.hasMinimumNegativeIndividuals():
            self.positivizeIndividuals()
        ###########################

        self.childrenCurrentGen = []
        self.parentsSelected = []
        #"###"+str(self.individuals[4]))

    def getBestNIndividuals(self, nIndiv):
        """Returns nIndiv (number) best individuals of current generation"""
        return self.individuals[0:nIndiv]

    def storeRule(self, rule: g3pIndividual):
        """Stores a rule (based on its support and confidence) in a set of rules"""
        if rule not in self.bestOverallIndividuals:
            #print(rule, rule.support, rule.confidence)
            self.bestOverallIndividuals.append(copy.deepcopy(rule))    
        
    
    def saveRule(self, rule: g3pIndividual):
        """Stores a rule (based on its fitness) in a text file if its not already in"""
        file = open(self.config.bestRulesFilePath, 'a')
        #if (str(rule)) not in fileContent:
        file.write(str(rule).strip() + " | Support: " + str(rule.support) + " | Confidence: " + str(rule.confidence) + " | Fitness: " + str(rule.fitness)
                   + " | AntcCovered: " + str(rule.antcCovered) + " | InstancesCovered: " + str(rule.instancesCovered) + "\n")
        # reversed = rule.reverseRule()
        
        # reversed = self.coverAll(reversed)
        # file.write("AntcCovered: "+str(reversed.antcCovered)+" RulesCovered: "+str(reversed.instancesCovered)+"\n")
        # del reversed
        file.flush()
        file.close()

    def removeClones(self):
        self.individuals = list(set(self.individuals))
        # Deterministic sort after set conversion to ensure reproducible results
        self.individuals.sort() 
        if len(self.individuals) <  self.populationSize:
           nNewIndividuals = self.populationSize-len(self.individuals)
           for i in range(nNewIndividuals):
               self.individuals.append(g3pIndividual(self.grammar.createSyntaxTree()))
               
    def positivizeIndividuals(self):
        newPositiveGen = [i if i.isPositiveCandidate() else i.reverseRule() for i in self.individuals]
        self.individuals = newPositiveGen

    def hasMinimumNegativeIndividuals(self):
        negatives = sum(not i.isPositiveCandidate() for i in self.bestOverallIndividuals)
        return negatives >= self.minimumNegativeIndividuals