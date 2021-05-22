#!/usr/bin/python3
from pandas import read_csv, DataFrame
from os import system

def getResources():
    return DataFrame(read_csv(r'resources/samples.csv'))

def getVHSDecodeCommand(args, sample):
    return "vhs-decode %s %s" % (args, sample)

def main():
    resources = getResources()
    for index, entry in resources.iterrows():
        if entry['Type'].upper() == ("VHS" or "UMATIC"):
            command = getVHSDecodeCommand(entry['Parameters'], entry['Sample Path']) + " testrun%04d" % index
            system(command)

if __name__ == '__main__':
    main()