
from functools import reduce
from evaluator import Evaluator
from g3pEngine import ClassificationStrategy, g3pEngine
import random
import numpy as np 

from sklearn.metrics import balanced_accuracy_score

from logger import log, logResult

class Fold:    
    def __init__(self):
        self.instances = []

    def setInstances(self, instances):
        self.instances = instances

    def getInstances(self):
        return self.instances

class CrossValidator:    
    def __init__(self, nFolds, dataset, kFold = True, progress_callback=None):
        self.nFolds = nFolds
        self.dataset = dataset
        self.folds = []
        self.kFold = kFold
        self.fitnessHistories = [] # To aggregate from all folds
        self.progress_callback = progress_callback

    def generateFolds(self):
        """Creates balanced nFolds folds"""
        self.folds = [Fold() for _ in range(self.nFolds)]
        positives = []
        negatives = []

        #split for unbalanced data
        for instance in self.dataset:
            if instance.getIsCandidate():
                positives.append(instance)
            else:
                negatives.append(instance)        

        if self.kFold:
            self.splitFolds(positives)
            self.splitFolds(negatives)
        else:
            self.splitHoldoutRep(positives)
            self.splitHoldoutRep(negatives)

    def splitFolds(self, instances):
        """Shuffles instances and assigns each instance to a fold"""
        #random.seed(terminalLogic.SEED)
        np.random.shuffle(instances)
        [self.folds[i % self.nFolds].getInstances().append(instances[i]) for i, _ in enumerate(instances)]

    def splitHoldoutRep(self, instances):
        """Shuffles instances and assigns each instance to a fold"""
        #random.seed(terminalLogic.SEED)
        np.random.shuffle(instances)
        [self.folds[i % self.nFolds].getInstances().append(instances[np.random.randint(0,len(instances))]) for i, _ in enumerate(instances)]

    def getTrainSet(self, i):
        train = []
        for j, _ in enumerate(self.folds):
            if j != i:
                train.extend(self.folds[j].getInstances())
        return train

    def getTestSet(self, i):
        test = []
        [test.append(instance) for instance in self.folds[i].getInstances()]
        return test

    
    def prune(self, inds, instances, threshold):
        """"Prunes inds, which are the selected rules for the classifier 
        using a threshold so that only the best rules are kept."""
        D = instances.copy()
        instancesToRemove = []
        coversToRemove = []
        rules = []
        coverFlag = False
        
        #Sort individuals
        inds = sorted(inds, key = lambda i: (i.fitness, i.confidence, i.support, i.getRuleSize()), reverse=True)
        
        #Set coverage counters
        C = []
        Caux = []
        [C.append(0) for _ in range(len(instances))]
        [Caux.append(0) for _ in range(len(instances))]

        for i in range(len(inds)): #For each rule/individual
            if len(D) == 0: break
            for j in range(len(D)): #For each instance/row in dataset
                antcResult, consqResult, ruleResult = g3pEngine.cover(D[j], inds[i])
                if antcResult:                    
                    Caux[j] = Caux[j]+1 #Increment coverage for this rule
                    if ruleResult:
                        coverFlag = True            
            if coverFlag:
                rules.append(inds[i])
                for j in range(len(C)):
                    C[j] = Caux[j]
                    if C[j] >= threshold:
                        instancesToRemove.append(D[j])
                        coversToRemove.append(j)
                
                Caux = self.reduceCoverageCounters(Caux, coversToRemove)
                C = self.reduceCoverageCounters(C, coversToRemove)
                D = list(set(D).difference(set(instancesToRemove)))
                instancesToRemove.clear()
                coversToRemove.clear()
            else:
                for j in range(len(C)):
                    Caux[j] = C[j]
            coverFlag = False
        
        return rules

    def customPrune(self, inds, instances, threshold):
        selectedRules = []
        uncoveredInstances = instances.copy()
        #Sort individuals
        inds = sorted(inds, key = lambda i: (i.fitness, i.confidence, i.support, i.getRuleSize()), reverse=True)
        
        for rule in inds:
            for instance in uncoveredInstances[:]: #We iterate through a copy to remove inside the loop
                antcResult, consqResult, ruleResult = g3pEngine.cover(instance, rule)
                if ruleResult:
                    if rule not in selectedRules:
                        selectedRules.append(rule)
                    uncoveredInstances.remove(instance)
            if len(uncoveredInstances) == 0: #everything is now covered
                break
        
        log(f"{len(selectedRules)} rules are selected")
        log(f"{len(uncoveredInstances)} instances are not covered")
        #selectedRules = sorted(selectedRules, key = lambda i: (-i.instancesCovered ,i.fitness, i.confidence, i.support, i.getRuleSize()), reverse=True)
        return selectedRules
        


    def trainFold(self,config, foldIdx, scoringBasedSorting = False, pause_event=None):
        """Starts the training process for a fold data and returns the selected rules after the training"""
        engine = g3pEngine(config, progress_callback=self.progress_callback, totalFolds=self.nFolds, foldIdx=foldIdx)
        engine.start(pause_event=pause_event)
        self.fitnessHistories.append(engine.fitnessHistory)
        #Remove duplicates
        engine.bestOverallIndividuals = list(set(engine.bestOverallIndividuals))

        rules = []
        rulesRest = []

        #To ensure having positive rules we first prune just with positive individuals.
        bestPositiveIndividuals = list(filter(lambda rule: rule.isPositiveCandidate(), engine.bestOverallIndividuals))       
        bestNegativeIndividuals = list(filter(lambda rule: not rule.isPositiveCandidate(), engine.bestOverallIndividuals))   
        rules = self.customPrune(bestPositiveIndividuals, engine.dataset, engine.pruneThreshold)
        rulesRest = self.customPrune(engine.bestOverallIndividuals, engine.dataset, engine.pruneThreshold)

        ######
        #rulesRest = engine.bestOverallIndividuals
        rulesRest.extend(rules)
        rules = list(set(rulesRest))

        #Now sort from more specific to less
        if not scoringBasedSorting:
            rules = sorted(rules, key = lambda i: (i.antcCovered,i.instancesCovered, -i.fitness))
        else: #Just for Scoring based CBA here!
            positiveRules = bestPositiveIndividuals
            negativeRules = bestNegativeIndividuals
            k = 3
            
            for rule in engine.bestOverallIndividuals:
                Wpos = sum((positive.confidence * positive.support) for positive in positiveRules)
                Wneg = sum((negative.confidence * negative.support) for negative in negativeRules)/k
                rule.score = (Wpos*rule.confidence+Wneg*(1-rule.confidence))/(Wpos+Wneg)
            rules = sorted(rules, key = lambda i: (i.score), reverse=True)

        log("Classifier rules:")
        for idx, rule in enumerate(rules):
            log(f"Rule {idx+1}: {str(rule).strip()} | Fitness: {rule.fitness:.4f} | Confidence: {rule.confidence:.4f}")

        [engine.saveRule(rule) for rule in rules]
        nPos = len(list(filter(lambda rule: rule.isPositiveCandidate(), rules)))
        log("|+rules=%d-rules=%d|" % (nPos,(len(rules)-nPos)))
        return rules

    def reduceCoverageCounters(self, counters, indexes):
        """Auxiliary method to assist pruning"""
        i = 0
        size = len(counters)
        indexesAux = indexes.copy()
        countersAux = counters.copy()
        
        while i < size:
            if i in indexesAux:
                countersAux.pop(i)
                indexesAux = indexesAux[1:]
                size -= 1
                indexesAux = [j-1 for j in indexesAux]
            else: 
                i += 1

        return countersAux

class Classifier:
    """Creates folds and configures the classifier strategy"""
    def __init__(self, dataset, nFolds, engineConfig, progress_callback=None):
        self.dataset = dataset
        self.crossValidator = CrossValidator(nFolds, dataset, progress_callback=progress_callback)
        self.crossValidator.generateFolds()
        self.engineConfig = engineConfig
        self.rulesFolds = []
        self.measures = []
        self.avgMeasures = []
        self.classifierStrategy = self.getClassificationStrategy()
        file = open(self.engineConfig.bestRulesFilePath, 'a')
        file.write('------\nNew experiment rules\n')
        file.flush()
        file.close()
    
    def train(self, pause_event=None):
        """Trains each fold individualy and appends resulting rules."""
        for i in range(len(self.crossValidator.folds)):
            # Wait if pause event is set
            if pause_event and pause_event.is_set():
                log(f"Experiment PAUSED after fold {i}. Waiting for resume...")
                while pause_event.is_set():
                    import time
                    time.sleep(1)
                log(f"Experiment RESUMED. Starting fold {i+1}...")

            log('Fold '+str(i))
            trainData = self.crossValidator.getTrainSet(i)
            self.engineConfig.dataset = trainData
            self.rulesFolds.append(self.crossValidator.trainFold(self.engineConfig, i, self.engineConfig.classificationStrategy == ClassificationStrategy.SCBA, pause_event=pause_event))

    def test(self):
        """Applies the testing phase of the classifier and takes all the measures for each fold."""
        #(acc, pre, rec, spe, f1, tp, fp, tn, fn)
        self.measures = [[0 for x in range(12)] for y in range(len(self.crossValidator.folds))] 
        avgAcc = 0
        precAcc = 0
        recallAcc = 0
        specificityAcc = 0
        balancedAvgAcc = 0
        balancedAvgAcc2 = 0
        for i in range(len(self.crossValidator.folds)):
            tp = 0
            tn = 0
            fp = 0
            fn = 0
            groundTruthValues = []
            predictions = []
            testData = self.crossValidator.getTestSet(i)
            for instance in testData:
                groundTruthValues.append(instance.getIsCandidate())
                if instance.getIsCandidate():
                    if self.classifierStrategy(self.rulesFolds[i], instance, len(testData)):
                        tp += 1
                        predictions.append(True)
                    else:
                        fn += 1
                        predictions.append(False)
                else:
                    if self.classifierStrategy(self.rulesFolds[i], instance, len(testData)):
                        fp += 1
                        predictions.append(True)
                    else:
                        tn += 1
                        predictions.append(False)
            try: 
                precisionAux = tp/(tp+fp)
                recallAux = tp/(tp+fn)
                specificityAux = tp/(tn+fp)
            except:
                precisionAux = 0.0
                recallAux = 0.0

            self.measures[i][0] += (tp+tn)/len(testData) if len(testData) > 0 else 0.0
            self.measures[i][1] += precisionAux
            self.measures[i][2] += recallAux

            avgAcc += self.measures[i][0]
            precAcc += self.measures[i][1]
            recallAcc += self.measures[i][2]

            try:
                self.measures[i][3] += tn/(tn+fp)
            except:
                self.measures[i][3] += 0
            specificityAcc += self.measures[i][3]
            try:
                self.measures[i][4] += 2*(precisionAux*recallAux)/(precisionAux+recallAux)
            except:
                self.measures[i][4] += 0
            self.measures[i][5] += tp
            self.measures[i][6] += fp
            self.measures[i][7] += tn
            self.measures[i][8] += fn

            #Balanced accuracy
            try:
                self.measures[i][9] = ((tp/(tp+fn))+(tn/(tn+fp)))/2
            except: 
                self.measures[i][9] = 0
            self.measures[i][10] = balanced_accuracy_score(groundTruthValues, predictions)
            balancedAvgAcc += self.measures[i][9]
            balancedAvgAcc2 += self.measures[i][10]

            log("Fold " + str(i))
            log("accuracy,precision,recall,specificity,fmeasure,tp,fp,tn,fn,balancedAcc")
            for j in range(len(self.measures[0])):
                if j < len(self.measures[0]) - 1:
                    log(str(self.measures[i][j])+',',True)
                else:
                    log(str(self.measures[i][j])+'\n',True)
        balancedAvgAcc = balancedAvgAcc/len(self.crossValidator.folds)
        avgAcc = avgAcc/len(self.crossValidator.folds)
        precAcc = precAcc/len(self.crossValidator.folds)
        recallAcc = recallAcc/len(self.crossValidator.folds)
        specificityAcc = specificityAcc/len(self.crossValidator.folds)
        balancedAvgAcc2 = balancedAvgAcc2/len(self.crossValidator.folds)

        log("Avg. accuracy: %f" % (avgAcc))
        log("Avg. precision: %f" % (precAcc))
        log("Avg. recall: %f" % (recallAcc))
        log("Avg. specificity: %f" % (specificityAcc))
        log("dataset;classStrategy;replStrategy;balAcc;accuracy;precision;recall;specificity;seed")
        logResult(f'{self.engineConfig.datasetFilePath};{self.classifierStrategy.__name__};{self.engineConfig.replacementStrategy};{balancedAvgAcc2};{avgAcc};{precAcc};{recallAcc};{specificityAcc};{self.engineConfig.seed}')
        self.avgMeasures.append(balancedAvgAcc2)
        self.avgMeasures.append(avgAcc)
        self.avgMeasures.append(precAcc)
        self.avgMeasures.append(recallAcc)
        self.avgMeasures.append(specificityAcc)
        return self.avgMeasures, self.crossValidator.fitnessHistories

    def calculateX2Independence(self, individual, numInstances):
        supp = individual.support
        conf = individual.confidence
        lift = individual.lift
        try:
            return numInstances*((lift-1)**2)*((supp*conf)/((conf-supp)*(lift-conf)))
        except:
            return 0

    def classifyCMAR(self, rulesFold, instance, trainSetSize):
        positiveRules = []
        negativeRules = []
        
        for rule in rulesFold:
            antcResult, consqResult, ruleResult = g3pEngine.cover(instance,rule)
            if antcResult:
                if ruleResult:
                    positiveRules.append(rule)
                else:
                    negativeRules.append(rule)

        if len(positiveRules) == 0 and len(negativeRules) == 0:
            return False
        elif len(positiveRules) > 0 and len(negativeRules) == 0:
            return True
        elif len(positiveRules) == 0 and len(negativeRules) > 0:
            return False
        else:
            wcsPositive = 0
            wcsNegative = 0
            for rule in positiveRules:
                chi = self.calculateX2Independence(rule, trainSetSize)   
                x2 = self.maxX2(rule, trainSetSize)
                if x2 != 0:
                    wcsPositive = wcsPositive + ((chi**2)/x2)         
            for rule in negativeRules:
                chi = self.calculateX2Independence(rule, trainSetSize)
                x2 = self.maxX2(rule, trainSetSize)
                if x2 != 0:
                    wcsNegative = wcsNegative + ((chi**2)/x2)  

            return wcsPositive>wcsNegative 
    
    def classifyOldCBA(self, rulesFold, instance, trainSetSize):        
        for rule in rulesFold:
            antcResult, consqResult, ruleResult = g3pEngine.cover(instance,rule)
            if antcResult: #if this rule covers this instance
                return self.getClassFromRule(rule)
        return False
    
    def classifyCBA(self, rulesFold, instance, trainSetSize):        
        for rule in rulesFold:
            antcResult, consqResult, ruleResult = g3pEngine.cover(instance,rule)
            if antcResult: #if this rule covers this instance
                return rule.syntaxTree[-1].symbol
        return False
    
    def classifyMultiStrategy(self, rulesFold, instance, trainSetSize):
        positiveRules = []
        negativeRules = []
        
        for rule in rulesFold:
            antcResult, consqResult, ruleResult = g3pEngine.cover(instance,rule)
            if antcResult:
                if ruleResult:
                    positiveRules.append(rule)
                else:
                    negativeRules.append(rule)

        if len(positiveRules) == 0 and len(negativeRules) == 0:
            return False
        elif len(positiveRules) > 0 and len(negativeRules) == 0:
            return True
        elif len(positiveRules) == 0 and len(negativeRules) > 0:
            return False
        else:
            return False
        
    def classifyCPAR(self, rulesFold, instance, trainSetSize):
        positiveRules = []
        negativeRules = []

        positiveFull = False
        negativeFull = False

        k = 5
        
        for rule in rulesFold:
            antcResult, consqResult, ruleResult = g3pEngine.cover(instance,rule)
            if antcResult:
                if rule.syntaxTree[-1].symbol and not positiveFull: #getClassFromRule 
                    positiveRules.append(rule)
                    if len(positiveRules) == k:
                        positiveFull = True
                elif not rule.syntaxTree[-1].symbol and not negativeFull:
                    negativeRules.append(rule)
                    if len(negativeRules) == k:
                        negativeFull = True

            if positiveFull and negativeFull:
                break

        positiveLaplace = sum(r.laplace for r in positiveRules)
        negativeLaplace = sum(r.laplace for r in negativeRules)

        if len(positiveRules) > 0 and len(negativeRules) == 0:
            return True
        elif len(positiveRules) == 0 and len(negativeRules) > 0:
            return False
        elif len(positiveRules) == 0 and len(negativeRules) == 0:
            return False
        else:
            return (positiveLaplace/len(positiveRules)) > (negativeLaplace/len(negativeRules))

    def maxX2(self, rule, nInstances):
        suppA = rule.suppAnt
        suppC = rule.suppCon
        try:
            e = (1/(suppA*suppC)) + (1/(suppA*(nInstances-suppC))) + (1/(suppC*(nInstances-suppA))) +(1/((nInstances-suppA)*(nInstances-suppC)));
            return (min(suppA, suppC)*(suppA*suppC/nInstances)**2)*nInstances*e 
        except:
            return 0

    def getClassFromRule(self, rule):
        return rule.syntaxTree.getClassFromRule()

    def getClassificationStrategy(self):
        if self.engineConfig.classificationStrategy == ClassificationStrategy.CBA:
            return self.classifyCBA
        elif self.engineConfig.classificationStrategy == ClassificationStrategy.CMAR:
             return self.classifyCMAR
        elif self.engineConfig.classificationStrategy == ClassificationStrategy.MULTISTRATEGY:
             return self.classifyMultiStrategy
        elif self.engineConfig.classificationStrategy == ClassificationStrategy.CPAR:
             return self.classifyCPAR
        elif self.engineConfig.classificationStrategy == ClassificationStrategy.SCBA:
             return self.classifyCBA
