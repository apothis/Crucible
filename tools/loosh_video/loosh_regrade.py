# Regrade a clean ChatGPT still to the Gym Membership sepia print look.
# usage: python loosh_regrade.py IN.webp OUT.webp [chroma shift]
# Method proven on GYM-07: LAB (signed a/b) chroma scale + warm mean shift toward reference stills,
# lifted blacks, vignette, grain, worn border lifted pixel-for-pixel from GYM-01's edges.
import sys, numpy as np
from PIL import Image, ImageCms, ImageFilter, ImageStat
S='loosh_gym_stills/'
src=Image.open(sys.argv[1]).convert('RGB'); W,H=src.size
chroma=float(sys.argv[3]) if len(sys.argv)>3 else 0.8; shift=float(sys.argv[4]) if len(sys.argv)>4 else 0.9
srgb=ImageCms.createProfile('sRGB'); lab=ImageCms.createProfile('LAB')
to_lab=ImageCms.buildTransform(srgb,lab,'RGB','LAB'); to_rgb=ImageCms.buildTransform(lab,srgb,'LAB','RGB')
def dec(im):
    a=np.asarray(ImageCms.applyTransform(im,to_lab)).astype(float); a[...,1:]=np.where(a[...,1:]>127,a[...,1:]-256,a[...,1:]); return a
def enc(a):
    o=a.copy(); o[...,0]=np.clip(o[...,0],0,255); o[...,1:]=np.clip(o[...,1:],-128,127)%256
    return ImageCms.applyTransform(Image.fromarray(o.astype('uint8'),'LAB'),to_rgb)
s=dec(src); sm=s.reshape(-1,3).mean(0)
rm=np.mean([dec(Image.open(S+f).convert('RGB')).reshape(-1,3).mean(0) for f in ['GYM-03_climb_a_stair.webp','GYM-06_one_pushup.webp','GYM-02_protein_cupboard.webp']],axis=0)
out=s.copy(); out[...,0]=s[...,0]*0.90+9
for c in (1,2): out[...,c]=(s[...,c]-sm[c])*chroma+sm[c]*(1-shift)+rm[c]*shift
g=np.asarray(enc(out)).astype(float)
yy,xx=np.mgrid[0:H,0:W]; d=np.sqrt(((xx-W/2)/(W/2))**2+((yy-H/2)/(H/2))**2)
g=g*np.clip(1-0.28*np.clip(d-0.55,0,None)/0.6,0.72,1)[...,None]+np.random.default_rng(7).normal(0,5.5,(H,W,3))
b=np.asarray(Image.open(S+'GYM-01_fitbit_pillow.webp').convert('RGB').resize((W,H))).astype(float); lum=b.mean(2); sat=b.max(2)-b.min(2)
edge=np.zeros((H,W),bool); e=70; edge[:e,:]=edge[-e:,:]=True; edge[:,:e]=edge[:,-e:]=True
ma=np.asarray(Image.fromarray((((lum>150)&(sat<90)&edge)*255).astype('uint8')).filter(ImageFilter.GaussianBlur(0.8))).astype(float)[...,None]/255
g=g*(1-ma)+b*ma
res=Image.fromarray(np.clip(g,0,255).astype('uint8')); res.save(sys.argv[2],quality=95)
rr,gg,bb=ImageStat.Stat(res).mean; print('out R/B %.2f sat %.0f'%(rr/bb, np.asarray(res.convert('HSV'))[...,1].mean()))
