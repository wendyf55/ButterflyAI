import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams['font.family']='DejaVu Sans'
W,H=160,84
fig,ax=plt.subplots(figsize=(18,8.4)); ax.set_xlim(0,W+1); ax.set_ylim(0,H); ax.axis('off')
TRAIN='#dcead5'; TEST='#fde8cc'; OOD='#e4dcf0'; MODEL='#d5e1ea'; EDGE='#6b6b6b'
bw,gap,x0=24.2,2.6,2.0
X=[x0+i*(bw+gap) for i in range(6)]
yb,hb=36,40
def box(x,y,w,h,c,lw=1.0):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=1.2',fc=c,ec=EDGE,lw=lw))
data=[
 (TRAIN,'BIMBY-BC2023',['iNaturalist, BIMBY project','BC, mostly Okanagan','Photos taken: 2023','4,091 real photos','2,137 F / 1,954 NF','Labels: BIMBY volunteers']),
 (TRAIN,'CPDA feeding composites',['BIMBY-BC2023 non-feeding','butterflies pasted onto','Okanagan flower photos','(iNaturalist API)','659 images, all NF','553 used in final model']),
 (OOD,'BIMBY-ON2023',['iNaturalist, BIMBY project','Ontario','Photos taken: 2023','2,785 photos','1,230 F / 1,555 NF','Labels: 4 students']),
 (TEST,'BIMBY-BC2024',['iNaturalist, BIMBY project','2,884 photos: BC, 2024','+ 1,307 photos of','unclear origin','1,950 copies of training','photos removed']),
 (TEST,'Joint gold standard',['iNaturalist, BIMBY project','97% BC','Photos taken: 2024','2,162 photos','10 target plants','Labels: feeding + plant']),
 (TRAIN,'Plant-species dataset',['iNaturalist API','Research Grade','North America, any year','10 target species','5,638 plain photos','14,046 CPDA composites']),
]
for x,(c,t,lines) in zip(X,data):
    box(x,yb,bw,hb,c)
    ax.text(x+bw/2,yb+hb-4.2,t,ha='center',va='center',fontsize=10.3,fontweight='bold',style='italic')
    for k,l in enumerate(lines):
        ax.text(x+1.6,yb+hb-10.5-k*4.8,l,ha='left',va='center',fontsize=10)
# models
my,mh=3,11
fx1,fx2=X[0],X[3]+bw; px1,px2=X[4],X[5]+bw
box(fx1,my,fx2-fx1,mh,MODEL,1.3); box(px1,my,px2-px1,mh,MODEL,1.3)
ax.text((fx1+fx2)/2,my+mh/2+1.8,'Butterfly-feeding model',ha='center',va='center',fontsize=13,fontweight='bold')
ax.text((fx1+fx2)/2,my+mh/2-2.4,'ResNet50 · feeding vs non-feeding',ha='center',va='center',fontsize=10)
ax.text((px1+px2)/2,my+mh/2+1.8,'Plant-species model',ha='center',va='center',fontsize=13,fontweight='bold')
ax.text((px1+px2)/2,my+mh/2-2.4,'ResNet50 · 10 plant species',ha='center',va='center',fontsize=10)
def arrow(x1,y1,x2,y2,ls='-'):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=16,lw=1.6,color='#444',linestyle=ls))
labels=['Training','Training','Out-of-region\ntest','Hold-out\ntest\n(Test 1)',None,'Training']
for i,(x,l) in enumerate(zip(X,labels)):
    cx=x+bw/2
    if i==4: continue
    arrow(cx,yb-0.5,cx,my+mh+0.5)
    ax.text(cx+1.2,(yb+my+mh)/2,l,ha='left',va='center',fontsize=10.5)
cx=X[4]+bw/2+5
arrow(cx,yb-0.5,cx,my+mh+0.5)
ax.text(cx+1.2,(yb+my+mh)/2,'Hold-out test\n(plant labels)',ha='left',va='center',fontsize=10.5)
arrow(X[4]+4,yb-0.5,X[3]+bw-6,my+mh+0.5,ls=(0,(4,3)))
ax.text(X[4]+5.2,yb-4.5,'feeding labels:\npart of Test 1',ha='left',va='center',fontsize=9.5,color='#333')
ax.text(158,79.3,'F = feeding, NF = non-feeding',ha='right',va='center',fontsize=10.5)
# legend
lx,ly=2,80
for k,(c,t) in enumerate([(TRAIN,'Training data'),(TEST,'Hold-out test data'),(OOD,'Out-of-region test data')]):
    box(lx+k*30,ly-2,3.2,2.6,c); ax.text(lx+k*30+4.2,ly-0.7,t,va='center',fontsize=10.5)
fig.savefig('dataset_sources_figure.png',dpi=220,bbox_inches='tight',facecolor='white')
fig.savefig('dataset_sources_figure.pdf',bbox_inches='tight',facecolor='white')
fig.savefig('dataset_sources_figure.svg',bbox_inches='tight',facecolor='white')
