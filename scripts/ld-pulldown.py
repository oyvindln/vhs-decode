# Extremely crude NTSC CAV pulldown script written in about a half hour, 
# POC code quality

import sys
import json

fname_in  = sys.argv[1]
fname_out = sys.argv[2]

with open(fname_in + '.json', 'r') as fp:
    in_json_str = fp.read()

in_json = json.loads(in_json_str)

out_json = {'videoParameters':    in_json['videoParameters'],
            'pcmAudioParameters': in_json['pcmAudioParameters']
           }


with open(fname_in + '.json', 'r') as fp:
    in_json_str = fp.read()

in_json = json.loads(in_json_str)

out_json = {'videoParameters':    in_json['videoParameters'],
            'pcmAudioParameters': in_json['pcmAudioParameters']
           }

tbc_fp_in  = open(fname_in,  'rb')
tbc_fp_out = open(fname_out, 'wb')

out_json['fields'] = []

fc = -1
prevf = None

for f in in_json['fields']:
    fdata = tbc_fp_in.read(910 * 263 * 2)
    #fdata = None 
    
    #print(f['vbi'])
    prevfc = fc
    
    if len(f['vbi']['vbiData']) >= 2 and f['vbi']['vbiData'][1] > 12*1024*1024:
        fc = 1
    else:
        fc += 1
        
    #print(fc)
        
    if fc <= 2:
        
        f['seqNo'] = len(out_json['fields']) + 1
        newIsFirstField = False if (f['seqNo'] % 2) else True
        
        #print(f['isFirstField'], newIsFirstField)
        
        if fc == 1:
            needFlip = not f['isFirstField']
            #if f['seqNo'] > (22047 * 2):
                #needFlip = f['isFirstField']
                
            if needFlip:
                #print(f['seqNo'] // 2)
                fdata2 = fdata
            else:
                fdata1 = fdata        
                
            f['isFirstField'] = True
        elif fc == 2:
            if needFlip:
                fdata1 = fdata
            else:
                fdata2 = fdata

            f['isFirstField'] = False
                
            tbc_fp_out.write(fdata1)
            tbc_fp_out.write(fdata2)
                
        # f['isFirstField'] = newIsFirstField
        out_json['fields'].append(f.copy())
        
    prevf = f
        

out_json['videoParameters']['numberOfSequentialFields'] = len(out_json['fields'])
with open(fname_out + '.json', 'w') as fp:
    fp.write(json.dumps(out_json))

