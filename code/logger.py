
#    This file is not part of FAST2.
#
#    Jose de la Torre Lopez, PhD Student
#    Knowledge and Discovery Systems (KDIS)
#    University of Cordoba, Spain

import threading
from datetime import datetime

_local = threading.local()

def setLogFile(path):
    _local.log_file = path

def getLogFile():
    return getattr(_local, 'log_file', 'log.txt')

startTime = datetime.now()

"""Message for the log"""
def log(msg, omitNewLine = False):
    outputFile = getLogFile()
    f = open(outputFile, 'a+')
    
    if not omitNewLine:
        content = str(datetime.now()) + ' | ' +  msg +'\n'
        f.write(content)
        print(content, end='')
    else:         
        f.write(msg)
        print(msg, end='')
    f.flush()
    f.close()

"""Logs a complete new line of results in the results file."""
def logResult(resultLine, omitTime = False):
    outputFile = "results.txt"
    f = open(outputFile, 'a+')   
    executionTime = (datetime.now() - startTime).total_seconds() / 60.0 
    if not omitTime:
        f.write(f'{resultLine};{executionTime}\n')
    else:
        f.write(f'{resultLine}\n')
    f.flush()
    f.close()

"""Logs a complete new line of results in the results file."""
def logStatistics(resultLine, omitTime = False):
    outputFile = "resultsStatistics.txt"
    f = open(outputFile, 'a+')   
    executionTime = (datetime.now() - startTime).total_seconds() / 60.0 
    if not omitTime:
        f.write(f'{resultLine};{executionTime}\n')
    else:
        f.write(f'{resultLine}\n')
    f.flush()
    f.close()

def setExecutionTimer():
    global startTime
    startTime = datetime.now()