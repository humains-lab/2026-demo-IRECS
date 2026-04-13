#    This file is not part of DEAP.
#
#    Rafael Barbudo Lunar, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

"""
The :mod:`g3p` for Grammar Guided Genetic Programming (G3P).

This module provides the methods and classes to perform G3P with :mod:`deap`.
It essentially contains the classes to build a gramatically based Genetic
Program Tree and the functions to evaluate it.
""" 

import copy
import random
import numpy as np 
# random.seed(1)  # Removed to avoid overriding user-specified seeds
import terminalLogic
from inspect import isclass, isfunction

#######################################
# G3P Data structure                  #
#######################################

class SyntaxTree(list):
    """
    Gramatically based genetic programming tree.

    Tree specifically formatted for optimization of G3P operations. The tree
    is represented with a list where the nodes (terminals and non-terminals)
    are appended in a depth-first order. The nodes appended to the tree are
    required to have an attribute *arity* which defines the arity of the
    primitive. An arity of 0 is expected from terminals nodes.
    """
    def __init__(self, content):
        list.__init__(self, content)

    def __deepcopy__(self, memo):
        new = self.__class__(self)
        new.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return new

    def __str__(self):
        """Return the syntax tree in a human readable string using infix notation."""
        output = ""
        for elem in self.getTerminals():
            output = output + elem.__str__() + " "
        return output.strip()

    def __repr__(self):
        return self.__str__()
    
    def getTerminals(self):
        terminals = []
        for elem in self:
            # Only terminals are considered. These terminals refers to both
            # terminals and arguments in the GP vocabulary
            if isinstance(elem, TerminalNode):
                terminals.append(elem)

        return terminals

    def searchSubtree(self, begin):
        """
        Return a slice object that corresponds to the
        range of values that defines the subtree which has the
        element with index *begin* as its root.
        """
        end = begin + 1
        total = self[begin].arity()
        while total > 0:
            total += self[end].arity() - 1
            end += 1
        return slice(begin, end)
    
    def getClassFromRule(self):
        genotype = self.getTerminals()
        return genotype[len(genotype)-3].symbol == 'equals'


class TerminalNode:
    """
    Terminal node of a SyntaxTree. It correspond to both primitives and
    arguments in GP. Terminal nodes have 0 arity and include the code being
    executed when the SyntaxTree is evaluated.
    """
    __slots__ = ('symbol', 'code', 'nOperators', 'type', 'minValue', 'maxValue')

    def __init__(self, symbol, code, nOperators = 0, type = "string", minValue=0, maxValue=4):
        self.symbol = symbol
        self.code = code
        self.nOperators = int(0 if nOperators is None else nOperators)
        self.type= type
        self.minValue = int(0 if minValue is None else minValue)
        self.maxValue = int(0 if maxValue is None else maxValue)
    
    def __str__(self):
        return str(self.symbol)
    
    def __repr__(self):
        return str(self.symbol)

    def arity(self):
        return 0


class NonTerminalNode:
    """
    Non-terminal node of a SyntaxTree. Each non-terminal node correspond
    to a production rule of a grammar. Thus, each node has a symbol
    (production rule left-sided) and the production itself
    (production rule right-sided)

    :Example:

    >>> <example> :: <a> <b> <c>
    self.symbol = example
    self.production = "a;b;c"
    self.prodList = ["a", "b", "c"]
    """
    __slots__ = ('symbol', 'production', 'prodList')

    def __init__(self, symbol, production):
        self.symbol = symbol
        # Python does not allow to use a list as the key of a dictionary
        self.production = production
        self.prodList = production.split(';')

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return str(self.symbol)
    
    def arity(self):
        return len(self.prodList)

    def getProduction(self, i):
        return self.prodList[i]


#######################################
# G3P Program generation functions    #
#######################################

class SyntaxTreeSchema:
    """
    Schema used to guarantee valid individuals are created. It constains the
    set of terminals and non-terminals that can be used to construct a
    SyntaxTree, among others. It also ensure the construction of trees with a
    bounded-size
    """
    __slots__ = ('pset', 'terminals', 'nonTerminals', 'maxDerivSize',
                 'minDerivSize', 'rootSymbol', 'terminalsMap',
                 'nonTerminalsMap', 'cardinalityMap')

    def __init__(self, maxDerivSize, rootSymbol, terminals,
                 nonTerminals):
        # Initialize the variables of the schema
        self.terminals = terminals
        self.nonTerminals = nonTerminals
        self.maxDerivSize = maxDerivSize
        self.minDerivSize = -1
        self.rootSymbol = rootSymbol
        self.terminalsMap = {}
        self.nonTerminalsMap = {}
        self.cardinalityMap = {}
        # Configure the schema
        self.configure()

    def setTerminalsDic(self):
        """Build and set the terminals dictionary."""
        self.terminalsMap = {}
        for termSymbol in self.terminals:
            self.terminalsMap[termSymbol.symbol] = termSymbol

    def setNonTerminalsDic(self):
        """Build and set the non terminals dictionary."""
        # Used to classify symbols
        auxMap = {}
        # Classify non-term symbols
        for nonTermSymbol in self.nonTerminals:
            nonTermSymbolName = nonTermSymbol.symbol
            if nonTermSymbolName in auxMap:
                auxMap.get(nonTermSymbolName).append(nonTermSymbol)
            else:
                lista = []
                lista.append(nonTermSymbol)
                auxMap[nonTermSymbolName] = lista

        # Create non-term symbols map
        self.nonTerminalsMap = {}
        for nonTermName in auxMap.keys():
            # Get symbols list
            lista = auxMap.get(nonTermName)
            # Put array in non terminals map
            self.nonTerminalsMap[nonTermName] = lista

    def setCardinalityDic(self):
        """
        Build and set the cardinality dictionary. This dictionary contains
        cardinality of all production rules (from cero to max number of
        derivations)
        """
        # Cardinality map
        self.cardinalityMap = {}
        for nonTermSymbol in self.nonTerminals:
            # Allocate space for cardinalities array
            list1 = [-1] * (1+self.maxDerivSize)
            # Put array in cardinality map
            self.cardinalityMap[nonTermSymbol.production] = list1

    def setMinDerivSize(self):
        """Calculate and set the minimum number of derivations."""
        for i in range(self.maxDerivSize):
            if self.symbolCardinality(self.rootSymbol, i) != 0:
                self.minDerivSize = i
                break

    def createSyntaxTree(self):
        """
        Create a new syntax tree of a random size in range
        (minDerivSize, maxDerivSize)

        :returns: SyntaxTree conformant with the grammar defined.
        """
        # Create resulting tree
        stree = SyntaxTree([])
        # Randomly determine the number of derivarion
        nOfDer = np.random.randint(self.minDerivSize, self.maxDerivSize)
        # Fill result branch
        self.fillTreeBranch(stree, self.rootSymbol, nOfDer)
        # Return resulting tree
        return stree

    def fillTreeBranch(self, tree, symbol, nOfDer):
        """
        Fill a SyntaxTree using the symbol and the allowed number of
        derivations

        :param symbol: The new symbol (terminal or non-terminal) to add
        :param nOfDer: The number of derivations
        """

        if symbol in self.terminalsMap:
            term = self.terminalsMap.get(symbol)
            if hasattr(terminalLogic, term.code) and str(term.code).__contains__("Value"): #Executes the function from code in terminalLogic
                tree.append(TerminalNode(getattr(terminalLogic, term.code)(term), term.code, term.nOperators, term.type, term.minValue, term.maxValue))
            else:
                tree.append(TerminalNode(symbol, term.code, term.nOperators, term.type))
        else:
            # Select a production rule
            selProd = self.selectProduction(symbol, nOfDer)

            if (selProd is not None):
                # Add this node
                tree.append(selProd)
                # Select a partition for this production rule
                selPart = self.selectPartition(selProd.prodList, nOfDer-1)
                # Apply partition, expanding production symbols
                selProdSize = len(selPart)

                for i in range(selProdSize):
                    self.fillTreeBranch(tree, selProd.prodList[i], selPart[i])
            else:
                self.fillTreeBranch(tree, symbol, nOfDer-1)

    def selectProduction(self, symbol, nOfDer):
        """
        Select a production rule for a symbol of the grammar, given the number
        of derivations available.

        :param symbol: Symbol to expand
        :param nOfDer: Number of derivations available

        :returns: A production rule for the given symbol or None if this symbol
        cannot be expanded using exactly such number of derivations
        """
        # Get all productions of this symbol
        prodRules = self.nonTerminalsMap.get(symbol)
        # Number of productions
        nOfProdRules = len(prodRules)
        # Create productions roulette
        roulette = [0] * nOfProdRules

        # Fill roulette
        try:
            for i in range(nOfProdRules):
                cardinalities = self.cardinalityMap.get(prodRules[i].production)
                # If this cardinality is not calculated, it will be calculated
                if cardinalities[nOfDer-1] == -1:
                    cardinalities[nOfDer-1] = self.pRuleDerivCardinality(
                        prodRules[i].prodList, nOfDer-1)
                    self.cardinalityMap[prodRules[i].production] = cardinalities

                roulette[i] = cardinalities[nOfDer-1]
                if i != 0:
                    roulette[i] = roulette[i] + roulette[i-1]

            # Choose a production at random
            randVal = roulette[nOfProdRules-1] * np.random.uniform(0, 1)

            for i in range(nOfProdRules):
                if randVal < roulette[i]:
                    return prodRules[i]
        except:  
            return None

    def selectPartition(self, prodRule, nOfDer):
        """
        Select a partition to expand a symbol using a production rule.

        :param prodRule: Production rule to expand
        :param nOfDer: Number of derivations available

        :returns: A partition
        """
        
        # Obtain all partitions for this production rule
        partitions = self.partitions(nOfDer, len(prodRule))
        # Number of partitions
        nOfPart = len(partitions)
        # Create partitions roulette
        roulette = [0] * nOfPart

        # Set roulette values
        for i in range(nOfPart):
            roulette[i] = self.pRulePartCardinality(prodRule, partitions[i])
            if i != 0:
                roulette[i] = roulette[i] + roulette[i-1]

        # Choose a production at random
        randVal = roulette[nOfPart-1] * np.random.uniform(0, 1)

        for i in range(nOfPart):
            if randVal < roulette[i]:
                return partitions[i]

        # This point shouldn't be reached
        return None

    def symbolCardinality(self, symbol, nOfDer):
        """
        Cardinality of a grammar symbol for the given number of derivs

        :param symbol: The grammar symbol (terminal or non-terminal)
        :param nOfDer: Number of derivations allowed

        :returns: Cardinality of the symbol
        """
        if symbol in self.terminalsMap:
            if nOfDer == 0:
                return 1
            else:
                return 0
        else:
            result = 0
            prodRules = self.nonTerminalsMap.get(symbol)
            for pRule in prodRules:
                cardinalities = self.cardinalityMap.get(pRule.production)
                if nOfDer <= 0:
                    result = result + self.pRuleDerivCardinality(
                        pRule.prodList, nOfDer-1)
                else:
                    # If this cardinality is not calculated, calculate it
                    if cardinalities[nOfDer-1] == -1:
                        cardinalities[nOfDer-1] = self.pRuleDerivCardinality(
                            pRule.prodList, nOfDer-1)
                        self.cardinalityMap[pRule.production] = cardinalities

                    result = result + cardinalities[nOfDer-1]
            return result

    def pRuleDerivCardinality(self, pRule, nOfDer):
        """
        Cardinality of a production rule for the given number of derivations.

        :param pRule: Production rule
        :param nOfDer: Number of derivations allowed

        :returns: Cardinality of the production rule
        """
        # Resulting cardinality
        result = 0
        # Obtain all partitions
        partitions = self.partitions(nOfDer, len(pRule))
        # For all partitions of nOfDer...
        for partition in partitions:
            result = result + self.pRulePartCardinality(pRule, partition)
        # Return result
        return result

    def pRulePartCardinality(self, prodRule, partition):
        """
        Cardinality of a production rule for the given partition.

        :param pRule: Production rule
        :param nOfDer: The given partition

        :returns: Cardinality of the production rule for the partition
        """
        prodRuleSize = len(prodRule)
        result = 1

        for i in range(prodRuleSize):
            factor = self.symbolCardinality(prodRule[i], partition[i])
            if factor == 0:
                return 0
            else:
                result = result * factor

        return result

    def partitions(self, total, dimension):
        result = []

        if dimension == 1:
            result.append([total])
        else:
            for i in range(total+1):
                pi = self.partitions(total-i, dimension-1)
                result = result + self.insertBefore(i, pi)

        return result

    def insertBefore(self, previous, strings):

        result = []
        for string in strings:
            tmp = [previous]
            for i in string:
                tmp.append(i)
            result.append(tmp)

        return result

    def configure(self):
        """
        Configure the different dictionaries and set the minimum
        derivation size given the set of terminals and non terminals
        """
        self.setTerminalsDic()
        self.setNonTerminalsDic()
        self.setCardinalityDic()
        self.setMinDerivSize()


#######################################
# G3P Crossovers                      #
#######################################

def cxSubtree(ind1, ind2):
    """
    Randomly select in each individual and exchange each subtree with the
    point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.

    :returns: A tuple of two trees.
    """
    nonTerms = []

    # Search non-terminals nodes
    for i in range(len(ind1)):
        if isinstance(ind1[i], NonTerminalNode):
            nonTerms.append(i)

    # Select one of the non-terminals
    point1 = nonTerms[np.random.randint(0, len(nonTerms))]

    # Look for the same non-terminal in the other individual
    point2 = -1
    for i in range(len(ind2)):
        try: 
            if(ind2[i].symbol == ind1[point1].symbol):
                point2 = i
                break
        except:
            point2 = -1
            break

    # If both individuals have the same non-terminal, then swap
    if point2 != -1:
        slice1 = ind1.searchSubtree(point1)
        slice2 = ind2.searchSubtree(point2)
        
        #slice1 = slice(slice1.start, slice1.stop + 1, slice1.step)
        #slice2 = slice(slice2.start, slice2.stop + 1, slice2.step)

        #if (slice1.stop - slice1.start - slice2.stop + slice2.start) == 0:
        #print("*****Cruce! Antes")
        #print(slice1,slice2)
        #print(str(ind1[slice1]),str(ind2[slice2])                                       )
        #print(str(ind1), str(ind2),sep="\n")
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]
        #print("*****Cruce! Despues")
        #print(str(ind1), str(ind2),sep="\n")
    return ind1, ind2


#######################################
# G3P Mutations                       #
#######################################

def mutSubtree(ind, schema):
    """
    Randomly select a point in the tree ind, then replace the
    subtree at that point as a root conformant to the restrictions
    imposed by schema.

    :param ind: The tree to be mutated.
    :param schema: SyntaxTreeSchema used to build the new subtree.

    :returns: A tuple of one tree.
    """
    # Select a non terminal symbol
    nonTerms = []

    for i in range(len(ind)):
        if isinstance(ind[i], NonTerminalNode):
            nonTerms.append(i)

    p0_branchStart = nonTerms[np.random.randint(0, len(nonTerms))]
    subtreeToMutate = ind.searchSubtree(p0_branchStart)
    p0_branchEnd = subtreeToMutate.stop

    # Assign the selected symbol
    selectedSymbol = ind[p0_branchStart]
    # Determine the maximum size to fill
    p0_swapBranch = 0

    for node in ind[subtreeToMutate]:
        if node.arity() != 0:
            p0_swapBranch = p0_swapBranch + 1

    # Save the fragment at the right of the subtree
    aux = []
    for i in range(p0_branchEnd, len(ind)):
        aux.append(ind[i])

    # Remove the subtree and the fragment at its right
    del ind[p0_branchStart:len(ind)]
    # Create son (second fragment) keeping the same number of derivations
    schema.fillTreeBranch(ind, selectedSymbol.symbol, p0_swapBranch)

    # Restore the fragment at the right of the subtree
    for node in aux:
        ind.append(node)

    # Return the mutated individual
    return ind