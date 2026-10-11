# Renders 3000x3000 DistroKid single covers for the finished Loosh tracks.
# Layout: the song's thumbnail still (crop inside the print border, 4x-UltraSharpV2 upscale
# on the BOX via ComfyUI, never on the Mac) on top, a dark band at the bottom with the
# artist name and title in Rye.
# Run from .mvwork with the repo venv, two steps:
#   ../.venv/bin/python ../tools/loosh_video/loosh_single_covers.py submit   (crop, upload, queue)
#   ../.venv/bin/python ../tools/loosh_video/loosh_single_covers.py compose  (fetch, letter, save)
from PIL import Image, ImageDraw, ImageFont
import hashlib, io, json, os, sys, urllib.request, urllib.parse, uuid

COMFY='http://192.168.1.201:8188'
FONT='fonts/Rye-Regular.ttf'; OUT='loosh_covers'
CREAM=(241,228,194,255); EMBER=(232,86,32,255); DARK=(30,18,10)
S=3000; IMG_H=2340                      # image area 3000x2340 (1.282:1), band 660
SONGS=[  # key, still, crop x0 (source px), title lines, line colours
 ('gym_membership','loosh_gym_stills/GYM-04_on_my_knees_thumbnail.webp',14,['SHE LEFT ME FOR A','GYM MEMBERSHIP'],[CREAM,EMBER]),
 ('beer_biscuits','loosh_bbbd_stills/BBBD-04_holy_trinity_thumbnail.webp',110,['BEER, BISCUITS,','& BAD DECISIONS'],[CREAM,EMBER]),
 ('heart_attack','loosh_hta_stills/HTA-04_call_my_mama_thumbnail.webp',14,['HONKY TONK','HEART ATTACK'],[CREAM,EMBER]),
 ('merica','loosh_mrc_stills/MRC-04_fireworks_on_a_tuesday_thumbnail.webp',14,["'MERICA, Y'ALL"],[EMBER]),
 ('three_beers','loosh_tbd_stills/TBD-04_straight_line_thumbnail.webp',14,['THREE BEERS DEEP'],[EMBER]),
]
Y0,Y1=14,1010                           # inside the rough print border
CW=round((Y1-Y0)*S/IMG_H)               # crop width for the 1.282 aspect
JOBS=f'{OUT}/_jobs.json'

def upload(png_bytes, name):
    b='----loosh'+uuid.uuid4().hex
    body=(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="{name}"\r\n'
          f'Content-Type: image/png\r\n\r\n').encode()+png_bytes+f'\r\n--{b}--\r\n'.encode()
    r=urllib.request.Request(f'{COMFY}/upload/image',body,{'Content-Type':f'multipart/form-data; boundary={b}'})
    return json.load(urllib.request.urlopen(r))['name']

def submit():
    os.makedirs(OUT,exist_ok=True); jobs={}
    for key,still,x0,*_ in SONGS:
        buf=io.BytesIO(); Image.open(still).convert('RGB').crop((x0,Y0,x0+CW,Y1)).save(buf,'PNG')
        data=buf.getvalue(); name=upload(data,f'loosh_cover_{hashlib.sha1(data).hexdigest()[:16]}.png')
        g={'1':{'class_type':'LoadImage','inputs':{'image':name}},
           '2':{'class_type':'UpscaleModelLoader','inputs':{'model_name':'4x-UltraSharpV2.safetensors'}},
           '3':{'class_type':'ImageUpscaleWithModel','inputs':{'upscale_model':['2',0],'image':['1',0]}},
           '4':{'class_type':'SaveImage','inputs':{'images':['3',0],'filename_prefix':f'loosh_cover_{key}'}}}
        r=urllib.request.Request(f'{COMFY}/prompt',json.dumps({'prompt':g}).encode(),{'Content-Type':'application/json'})
        jobs[key]=json.load(urllib.request.urlopen(r))['prompt_id']; print(key,'queued')
    json.dump(jobs,open(JOBS,'w'))

def compose():
    jobs=json.load(open(JOBS))
    for key,still,x0,lines,colors in SONGS:
        h=json.load(urllib.request.urlopen(f'{COMFY}/history/{jobs[key]}'))
        if jobs[key] not in h: print(key,'not finished'); continue
        im=h[jobs[key]]['outputs']['4']['images'][0]
        q=urllib.parse.urlencode({'filename':im['filename'],'subfolder':im['subfolder'],'type':im['type']})
        up=Image.open(io.BytesIO(urllib.request.urlopen(f'{COMFY}/view?{q}').read())).convert('RGB')
        cover=Image.new('RGB',(S,S),DARK); cover.paste(up.resize((S,IMG_H),Image.LANCZOS),(0,0))
        # feather the image into the band
        fade=Image.new('L',(1,260)); fade.putdata([int(255*(i/259)**1.6) for i in range(260)])
        cover.paste(Image.new('RGB',(S,260),DARK),(0,IMG_H-260),fade.resize((S,260)))
        d=ImageDraw.Draw(cover,'RGBA')
        def centred(t,f,y,col,stroke):
            bb=d.textbbox((0,0),t,font=f); x=(S-(bb[2]-bb[0]))//2-bb[0]
            d.text((x+8,y+8-bb[1]),t,font=f,fill=(0,0,0,200))
            d.text((x,y-bb[1]),t,font=f,fill=col,stroke_width=stroke,stroke_fill=DARK)
            return bb[3]-bb[1]
        # title size: largest that fits 2760 px wide and the band height
        size=300
        while True:
            ft=ImageFont.truetype(FONT,size)
            if max(d.textbbox((0,0),l,font=ft)[2] for l in lines)<=2760 and size*len(lines)*1.15<=470: break
            size-=10
        fa=ImageFont.truetype(FONT,120)
        hgt=lambda t,f: d.textbbox((0,0),t,font=f)[3]-d.textbbox((0,0),t,font=f)[1]
        block=hgt('LOOSH WOGGINS',fa)+60+sum(hgt(l,ft)+40 for l in lines)-40
        y=IMG_H-110+(S-(IMG_H-110)-block)//2      # centre the text block in the band
        y+=centred('LOOSH WOGGINS',fa,y,CREAM,6)+60
        for l,c in zip(lines,colors): y+=centred(l,ft,y,c,10)+40
        cover.save(f'{OUT}/Loosh_Woggins_{key}_cover_3000.jpg',quality=95)
        print(key,'written', up.size, 'title size',size)

{'submit':submit,'compose':compose}[sys.argv[1]]()
