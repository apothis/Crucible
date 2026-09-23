# Renders caption + branding PNG overlays for the Gym Membership lyric video and a thumbnail.
# Writes loosh_gym_build/ov_*.png and loosh_gym_build/overlays.txt (file,start,end).
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import csv, os
B='loosh_bbbd_build'; FONT='fonts/Rye-Regular.ttf'; W,H=1920,1080
CREAM=(241,228,194,255); EMBER=(232,86,32,255); DARK=(43,26,14,230)
def text_png(lines, sizes, y_anchor='bottom', pad_bottom=150, color=CREAM, tracking=0, opacity=255):
    im=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(im)
    fonts=[ImageFont.truetype(FONT,s) for s in sizes]
    heights=[d.textbbox((0,0),l,font=f)[3]-d.textbbox((0,0),l,font=f)[1] for l,f in zip(lines,fonts)]
    gap=14; total=sum(heights)+gap*(len(lines)-1)
    y = H-pad_bottom-total if y_anchor=='bottom' else (H-total)//2 if y_anchor=='center' else pad_bottom
    for l,f,h in zip(lines,fonts,heights):
        bb=d.textbbox((0,0),l,font=f); w=bb[2]-bb[0]; x=(W-w)//2-bb[0]; yy=y-bb[1]
        d.text((x+4,yy+4),l,font=f,fill=(0,0,0,190))                     # shadow
        d.text((x,yy),l,font=f,fill=color,stroke_width=3,stroke_fill=DARK) # face + dark stroke
        y+=h+gap
    if opacity<255:
        a=im.split()[3].point(lambda v: v*opacity//255); im.putalpha(a)
    return im
out=[]; n=0
def emit(im,s,e):
    global n; p=f'{B}/ov_{n:02d}.png'; im.save(p); out.append((p,s,e)); n+=1
# captions
for line in open('loosh_bbbd_captions.csv'):
    if line.startswith('#') or not line.strip(): continue
    s,e,size,text=line.rstrip('\n').split(',',3)
    s,e=float(s),float(e)
    if s>=157: e=min(e,181.5)                     # make room for the end card
    fs=64 if size=='big' else 46
    import textwrap
    lines=textwrap.wrap(text, 46 if size=='big' else 64)
    emit(text_png(lines,[fs]*len(lines)),s,e)
# branding: title card over the intro, wordmark bug, end card
# intro title card dropped: the album cover card is the title card for this song
bug=Image.new('RGBA',(W,H),(0,0,0,0)); d=ImageDraw.Draw(bug); f=ImageFont.truetype(FONT,30)
t='LOOSH WOGGINS'; bb=d.textbbox((0,0),t,font=f); x=W-(bb[2]-bb[0])-48; y=40
d.text((x+2,y+2),t,font=f,fill=(0,0,0,150)); d.text((x,y),t,font=f,fill=(241,228,194,170))
p=f'{B}/ov_{n:02d}.png'; bug.save(p); out.append((p,12.5,181.5)); n+=1
emit(text_png(['LOOSH WOGGINS','Beer, Biscuits & Bad Decisions','Real Country. Real Bass. Real Questionable.'],[96,48,34],'center'),182.0,187.9)
with open(f'{B}/overlays.txt','w') as fh:
    for p,s,e in out: fh.write(f'{p},{s},{e}\n')
print(n,'overlays written')
# thumbnail 1280x720 from GYM-04, title in the right third
src=Image.open('loosh_bbbd_stills/BBBD-04_holy_trinity_thumbnail.webp').convert('RGB').crop((0,80,1536,944)).resize((1280,720),Image.LANCZOS)
th=Image.new('RGBA',(1280,720),(0,0,0,0)); d=ImageDraw.Draw(th)
fb=ImageFont.truetype(FONT,34); ft=ImageFont.truetype(FONT,78)
def draw_right(lines,fonts,y0,gap=10,colors=None):
    y=y0
    for i,(l,f) in enumerate(zip(lines,fonts)):
        bb=d.textbbox((0,0),l,font=f); w=bb[2]-bb[0]; x=1280-40-w-bb[0]
        d.text((x+4,y+4),l,font=f,fill=(0,0,0,200))
        d.text((x,y),l,font=f,fill=(colors[i] if colors else CREAM),stroke_width=4,stroke_fill=DARK)
        y+=bb[3]-bb[1]+gap
draw_right(['LOOSH WOGGINS'],[fb],150)
draw_right(['BEER, BISCUITS','& BAD','DECISIONS'],[ft,ft,ft],200,colors=[CREAM,EMBER,CREAM])
thumb=Image.alpha_composite(src.convert('RGBA'),th).convert('RGB')
thumb.save('loosh_channel/thumbnail_beer_biscuits_1280x720.jpg',quality=92)
print('thumbnail written')
