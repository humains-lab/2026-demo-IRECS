import xml.etree.ElementTree as ET
import os
from g3p import TerminalNode, NonTerminalNode

def parse(filename):
    # Parse the file and get the starting point
    filename = str(filename)
    code_dir = os.path.abspath(os.path.dirname(__file__))

    if os.path.isabs(filename):
        grammar_path = filename
    else:
        grammar_path = os.path.abspath(os.path.join(code_dir, filename))

    root = ET.parse(grammar_path).getroot()

    # Parse the root symbol
    rootSymbol = root.find('root-symbol').text

    terminals = []

    for tElem in root.findall('terminals/terminal'):
        tName = tElem.get('name')
        tCode = tElem.get('code')
        tNOperators = tElem.get('nOperators')
        tType = tElem.get('type')
        tMaxValue = tElem.get('maxValue')
        tMinValue = tElem.get('minValue')
        terminals.append(TerminalNode(tName, tCode, tNOperators, tType, tMinValue, tMaxValue))

    # Parse the non terminal nodes (i.e. the production rules)
    nonTerminals = []
    for ntElem in root.findall('non-terminals/non-terminal'):
        name = ntElem.get('name')
        for production in ntElem.findall('production-rule'):
            nonTerminals.append(NonTerminalNode(name, production.text))


    return rootSymbol, terminals, nonTerminals   

def printGrammar(rootSymbol, terminals, nonTerminals):
    print("Root symbol: " + rootSymbol)
    print("Terminals")
    for t in terminals: 
        print(t)
    print("Non-terminals")
    for n in nonTerminals:
        print(n.symbol)
        print(n.prodList)

def printProdList(nonTerminals):
    for n in nonTerminals:
        print(n.symbol + " => ", end = '')
        print(n.prodList)
