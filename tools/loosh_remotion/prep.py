# Build public/data/<song>.json from the .mvwork cue/caption/word files.
# usage: python prep.py bbbd
import csv, json, sys, re
song=sys.argv[1]; MV='/Volumes/SSD1/Code/MusicGen/.mvwork/'
cues=[]; drops=[]
for line in open(MV+f'loosh_{song}_cues.csv'):
    if line.startswith('# waveform strip'):
        drops=[[float(a),float(b)] for a,b in re.findall(r'([\d.]+)-([\d.]+)',line)]
        continue
    if line.startswith('#') or line.startswith('start,') or not line.strip(): continue
    s,e,sec,still,move,src=line.strip().split(',')
    cues.append({'start':float(s),'end':float(e),'section':sec,'still':still+'.webp','move':move})
sizes=[r.split(',')[2] for r in open(MV+f'loosh_{song}_captions.csv') if not r.startswith('#') and r.strip()]
words=json.load(open(MV+f'loosh_{song}_build/words.json'))
assert len(sizes)==len(words),(len(sizes),len(words))
for l,sz in zip(words,sizes): l['size']=sz
out={'fps':25,'duration':cues[-1]['end'],'audio':f'audio/{song}.wav','sections':cues,'lines':words,'drops':drops,
     'title':'Beer, Biscuits, & Bad Decisions','artist':'LOOSH WOGGINS','album':'Beer, Biscuits & Bad Decisions',
     'motto':'Real Country. Real Bass. Real Questionable.',
     'short':{'start':41.5,'cut':54.8,'end':69.0,'stills':['BBBD-04p_holy_trinity_portrait.webp','BBBD-05p_flatbed_drop_portrait.webp']}}
json.dump(out,open(f'public/data/{song}.json','w'),indent=1)
print(len(cues),'sections',len(words),'lines',len(drops),'drop windows, duration',out['duration'])
