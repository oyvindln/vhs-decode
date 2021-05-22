#!/usr/bin/python3
from pandas import read_csv, DataFrame
from os import system, path, getcwd, chdir
import logging

CURRENT_DIR = getcwd()
WORKING_DIR = '~/vault/vhsdecode_runs'
HOME = path.expanduser("~")
DROPOUT_COMPENSATE = True
DRY_RUN = False
DEMODTHREADS = 4
LOGFILE = "batch_tests.log"

logging.basicConfig(
    filename=LOGFILE,
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger()

def pathLeaf(ospath):
    head, tail = path.split(ospath)
    return tail or path.basename(head)

def fullPath(short):
    return short.replace('~', HOME)

def getResources():
    return DataFrame(read_csv(r'resources/samples.csv'))

def getVHSFlags(standard, args):
    if standard.upper() == "PALM":
        standard_flags = "-pm"
    elif standard.upper() == "NTSC":
        standard_flags = "-n"
    else:
        standard_flags = "-p"

    if DROPOUT_COMPENSATE:
        standard_flags = "%s %s" % (standard_flags, "--doDOD")

    try:
        if args.isdigit():
            args = ""
    except AttributeError:
        args = ""

    demod_t = "-t %d" % DEMODTHREADS

    return "%s %s %s" % (standard_flags, args, demod_t)

def getVHSDecodeCommand(type, standard, args, sample):
    if type.upper() == "VHS":
        return "vhs-decode %s \"%s\"" % (getVHSFlags(standard, args), sample)
    else:
        return "vhs-decode -U %s \"%s\"" % (getVHSFlags(standard, args), sample)

def getGENCommand(script, sample):
    return "%s %s" % (script, sample)

def getOUTFileName(index):
    return "testrun%04d" % index

def moveToWorkDir():
    if path.isdir(fullPath(WORKING_DIR)):
        chdir(fullPath(WORKING_DIR))
    else:
        print('Destination directory %s not found!' % WORKING_DIR)
        exit(0)

def returnAndExit():
    chdir(CURRENT_DIR)

def decodeLoop(resources):
    for index, entry in resources.iterrows():
        if entry['Type'].upper() == "VHS" or entry['Type'].upper() == "UMATIC":
            command = '%s %s' % (
                getVHSDecodeCommand(
                    entry['Type'],
                    entry['Standard'],
                    entry['Parameters'],
                    fullPath(entry['Sample Path'])
                ),
                getOUTFileName(index)
            )
            logger.info('Executing: %s' % command)
            logger.info('Decoding: %s' % pathLeaf(entry['Sample Path']))
            if not DRY_RUN:
                system(command)
        else:
            logger.warning('Ignoring: %s' % entry['Sample Path'])

def genLoop(resources):
    for index, entry in resources.iterrows():
        logger.info('Generating: %s.mkv' % getOUTFileName(index))
        command = getGENCommand(entry['ChromaScript'], getOUTFileName(index))
        logger.info('Executing: %s' % command)
        if not DRY_RUN:
            system(command)

def main():
    resources = getResources()
    moveToWorkDir()
    decodeLoop(resources)
    genLoop(resources)
    returnAndExit()

if __name__ == '__main__':
    main()