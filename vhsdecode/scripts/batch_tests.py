#!/usr/bin/python3
from pandas import read_csv, DataFrame
from os import system, path, getcwd, chdir
import logging
import argparse
import subprocess
from datetime import datetime

CURRENT_DIR = getcwd()
WORKING_DIR = '~/vault/vhsdecode_runs'
HOME = path.expanduser("~")
DROPOUT_COMPENSATE = True
DRY_RUN = False
DEMODTHREADS = 4
LOGFILE = "batch_tests.log"
RESOURCES_FILE = 'resources/samples.csv'
TTF_FONT = path.abspath('resources/Vintage2513ROM.ttf')
FONT_SIZE = 32
SUBTITLES_POSITION = 110, 220  # x and y position of the test chart subtitles
BACKGROUND_IMAGE = path.abspath('resources/PAL_testchart.png')
PRESENTATION_CHART = 'presentation.mkv'
GREETING_ENDING_CHART = 'ending_greeting.mkv'

logging.basicConfig(
    filename=LOGFILE,
    filemode='w',
    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger()

def getCmdArgs():
    parser = argparse.ArgumentParser(description='Run the decoding toolchain against RF samples')
    parser.add_argument('-r', '--resources', nargs='?',
                        default=RESOURCES_FILE, type=str,
                        help='specify resource list file (default: %s)' % RESOURCES_FILE)
    parser.add_argument('-o', '--output_dir', nargs='?',
                        default=WORKING_DIR, type=str,
                        help='specify output directory (default: %s)' % WORKING_DIR)
    parser.add_argument('-wn', '--welcome_notes', nargs='?',
                        default=WORKING_DIR, type=str,
                        help='Adds notes to the first video chart')
    group = parser.add_argument_group("Decoding")
    group.add_argument('-t', '--demod_threads', nargs='?',
                        default=-1, type=int,
                        help='Demodulator threads')
    group.add_argument('-nd', '--noDOD', action='store_true',
                       help="Don't compensate dropouts")
    group.add_argument('-xf', '--extra_flags', nargs='?',
                        default='', type=str,
                        help='Add extra global single decoding flags (example -xf="-ct --noAGC")')
    group.add_argument('-n', '--dry-run', action='store_true',
                       help='Show commands, rather than running them')
    group.add_argument('-g', '--gen-vid', action='store_true',
                       help='Run video generation scripts only')
    group.add_argument('-a', '--assemble', action='store_true',
                       help='Go directly to assemble the final demo testing video with charts')
    group.add_argument('-l', '--list', action='store_true',
                       help='List samples and exit')
    group.add_argument('-s', '--skip', nargs='?',
                        default=-1, type=int,
                        help='Skip first N samples on the list')


    return parser.parse_args()

def pathLeaf(ospath):
    head, tail = path.split(ospath)
    return tail or path.basename(head)

def fullPath(short):
    return short.replace('~', HOME)

def getResources(resources):
    return DataFrame(read_csv(resources))

def cleanParams(params):
    try:
        if params.isdigit():
            params = ""
    except AttributeError:
        params = ""
    return params

def paramsCompose(params):
    if len(args.extra_flags) > 0:
        split_params = params.split()
        extra_params = args.extra_flags.split()
        params_diff = [item for item in extra_params if item not in split_params]
        for diff in params_diff:
            split_params.append(diff)
        return ' '.join(split_params)
    else:
        return params

def getVHSFlags(standard, params):
    if standard.upper().strip() == "PALM":
        standard_flags = "-pm"
    elif standard.upper().strip() == "NTSC":
        standard_flags = "-n"
    else:
        standard_flags = "-p"

    if DROPOUT_COMPENSATE:
        standard_flags = "%s %s" % (standard_flags, "--doDOD")

    params = cleanParams(params)
    demod_t = "-t %d" % DEMODTHREADS

    return paramsCompose("%s %s %s" % (standard_flags, params, demod_t))

def isVHSVariant(type):
    return type.upper().strip()[-3:] == "VHS"

def isSVHS(type):
    return type.upper().strip() == "SVHS" or type.upper().strip() == "S-VHS"

def isVHSDecodable(type):
    return isVHSVariant(type) or \
        type.upper().strip() == "UMATIC"

def getVHSDecodeCommand(type, standard, args, sample):
    if isVHSVariant(type) and not isSVHS(type):
        return "vhs-decode %s \"%s\"" % (getVHSFlags(standard, args), sample.strip())
    elif isSVHS(type):
        return "vhs-decode -tf SVHS %s \"%s\"" % (getVHSFlags(standard, args), sample.strip())
    else:
        return "vhs-decode -U %s \"%s\"" % (getVHSFlags(standard, args), sample.strip())

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
        if index > args.skip:
            if isVHSDecodable(entry['Type']):
                command = '%s %s' % (
                    getVHSDecodeCommand(
                        entry['Type'],
                        entry['Standard'],
                        entry['Parameters'],
                        fullPath(entry['Sample Path'].strip())
                    ),
                    getOUTFileName(index)
                )
                logger.info('Executing: %s' % command)
                logger.info('Decoding: %s' % pathLeaf(entry['Sample Path']))
                if not DRY_RUN:
                    system(command)
                else:
                    print(command)
            else:
                logger.warning('Ignoring: %s' % entry['Sample Path'])

def genLoop(resources):
    for index, entry in resources.iterrows():
        if index > args.skip:
            logger.info('Generating: %s.mkv' % getOUTFileName(index))
            command = getGENCommand(entry['ChromaScript'], getOUTFileName(index))
            logger.info('Executing: %s' % command)
            if not DRY_RUN:
                system(command)
            else:
                print(command)

def getGitCommit():
    run = subprocess.run(['git', 'log', '-1', '--oneline', '--no-decorate'], stdout=subprocess.PIPE)
    logparts = run.stdout.split()
    return logparts[0].decode("utf-8").strip()

def getGitCommitDate():
    run = subprocess.run(['git', 'log', '-1', '--format=%cd', '--date=iso'], stdout=subprocess.PIPE)
    return run.stdout.decode("utf-8").strip()

def getGitBranch():
    run = subprocess.run(['git', 'branch', '--show-current'], stdout=subprocess.PIPE)
    return run.stdout.decode("utf-8").strip()

def getFrameCount(video):
    run = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video],
        stdout=subprocess.PIPE
    )
    return run.stdout.decode("utf-8").strip()

def getGitInfo():
    date = datetime.strptime(getGitCommitDate(), '%Y-%m-%d %H:%M:%S %z')
    return {
        'commit': getGitCommit(),
        'date': date.strftime('%b %d %Y'),
        'branch': getGitBranch()
    }

def osdText(x, y, text, size=FONT_SIZE, inter=10):
    lines = text.split('\n')
    ypos = y
    step = size + inter
    dt_assy = ""
    for line in lines:
        drawtext = "fontfile='%s': text='%s': fontcolor=white: fontsize=%d: x=%d: y=%d" % \
                   (TTF_FONT, line, size, x, ypos)
        ypos += step
        dt_assy += "drawtext=%s," % drawtext
    return "-vf \"fps=25,%s\" " % dt_assy.strip(',')

def genInitSlide(git_info):
    ffmpeg_base = "ffmpeg -y -r 1/5 -i \"%s\" -c:v libx264 -crf 0 -vf \"fps=25,format=yuv420p\" " % BACKGROUND_IMAGE
    if len(args.welcome_notes) == 0:
        text = "Test run %s\n- Branch %s\n- Commit %s" % (
            git_info['date'], git_info['branch'], git_info['commit']
        )
    else:
        text = "Test run %s\n- Branch %s\n- Commit %s\n%s" % (
            git_info['date'], git_info['branch'], git_info['commit'], args.welcome_notes.strip()
        )
    command = ffmpeg_base + osdText(SUBTITLES_POSITION[0], SUBTITLES_POSITION[1], text)
    command += " %s" % PRESENTATION_CHART
    logger.info('Executing: %s' % command)
    if DRY_RUN:
        print(command)
    else:
        system(command)

def autoClipText(text, length=28, tail=5):
    separator="~"
    span = length - tail - len(separator)
    if len(text) > length:
        return text[:span] + "~" + text[-tail:]
    else:
        return text

def autoLineParams(params, length=20):
    split_params = params.split()
    separator = ' '
    join_params = separator.join(split_params)
    if len(join_params) > length:
        start = 0
        assy = ''
        for param in split_params:
            assy += "%s " % param
            if len(assy) - start > length:
                start = len(assy)
                assy = assy.strip() + '\n'
        join_params = assy.strip()
    return join_params

def genSampleSlide(type, samplefile, standard, parameters, notes, framecount, outname):
    sample = autoClipText(samplefile)
    params = autoLineParams(parameters)
    ffmpeg_base = "ffmpeg -y -r 1/3 -i \"%s\" -c:v libx264 -crf 0 -vf \"fps=25,format=yuv420p\" " % BACKGROUND_IMAGE

    if len(cleanParams(notes)) == 0:
        text = "%s %s, %s frames\n%s\n%s" % \
               (standard, type, framecount, sample, params)
    else:
        text = "%s %s %s\n%s\n%s\n%s" % \
               (standard, type, framecount, sample, params, notes)

    command = ffmpeg_base + osdText(SUBTITLES_POSITION[0], SUBTITLES_POSITION[1], text)
    command += " %s.mkv" % outname
    logger.info('Executing: %s' % command)
    if DRY_RUN:
        print(command)
    else:
        system(command)

def genSampleSlides(resources):
    for index, entry in resources.iterrows():
        if index > args.skip:
            if isVHSDecodable(entry['Type']):
                genSampleSlide(
                    entry['Type'].upper().strip(),
                    pathLeaf(entry['Sample Path'].strip()),
                    entry['Standard'].upper().strip(),
                    getVHSFlags(entry['Standard'].upper().strip(), entry['Parameters']),
                    cleanParams(entry['Notes']),
                    getFrameCount("%s.mkv" % getOUTFileName(index)),
                    "slide_%s" % getOUTFileName(index)
                )

def genGreetings():
    ffmpeg_base = "ffmpeg -y -r 1/5 -i \"%s\" -c:v libx264 -crf 0 -vf \"fps=25,format=yuv420p\" " % BACKGROUND_IMAGE
    text = "Thanks for watching\\!\n\nSee description for more info."
    command = ffmpeg_base + osdText(SUBTITLES_POSITION[0], SUBTITLES_POSITION[1], text)
    command += " %s" % GREETING_ENDING_CHART
    logger.info('Executing: %s' % command)
    if DRY_RUN:
        print(command)
    else:
        system(command)


def listSamples(resources):
    for index, entry in resources.iterrows():
        print(
            "%04d" % index,
            entry['Type'].upper().strip(),
            entry['Standard'].upper().strip(),
            pathLeaf(entry['Sample Path'].strip())
        )

def main():
    resources = getResources(args.resources)
    if not args.list:
        git_info = getGitInfo()
        moveToWorkDir()
        if not args.gen_vid and not args.assemble:
            decodeLoop(resources)
        if not args.assemble:
            genLoop(resources)
        genInitSlide(git_info)
        genSampleSlides(resources)
        genGreetings()
        returnAndExit()
    else:
        listSamples(resources)

if __name__ == '__main__':
    args = getCmdArgs()
    if not DRY_RUN:
        DRY_RUN = args.dry_run
    if args.demod_threads > 0:
        DEMODTHREADS = args.demod_threads
    if args.noDOD:
        DROPOUT_COMPENSATE = False
    WORKING_DIR = args.output_dir
    main()
