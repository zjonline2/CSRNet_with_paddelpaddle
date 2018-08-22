import os
import glob
import json
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from PIL import Image
import pp
job_server = pp.Server() 
def distance(a,b):
    if a.has_key('w') and b.has_key('h'):
       a=[(a['x']+a['w'])*0.5,(a['y']+a['h'])*0.5]
       b=[(b['x']+b['w'])*0.5,(b['y']+b['h'])*0.5]
    elif a.has_key('x') and b.has_key('y'):
       a=[a['x'],a['y']]
       b=[b['x'],b['y']]
    return ((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5
img_paths =[];
path='./image/stage2/train/'
annotations=json.load(open('./annotation/annotation_train_stage2.json'))['annotations']
id=np.zeros([3000])
gt=os.listdir('./ground_truth/')
for i in gt:
    id[int(i.replace('.npy',''))]=1
def process(dy,di):
    from scipy.ndimage.filters import gaussian_filter
    return gaussian_filter(dy,di, mode='constant')
def sum(gt):
    s=0
    for x in range(gt.shape[0]):
        for y in range(gt.shape[1]):
            s+=gt[x,y]
    return s
def main(annotations,id):
    import json
    import numpy as np
    
    from PIL import Image
    belta=0.3
    for annotation in annotations:
      if not id[int(annotation['id'])]==1:
        img=Image.open('./image/'+annotation['name']);ps=annotation['annotation'];distances=[[] for j in ps]
        img=np.array(img);
        density = np.zeros([img.shape[1],img.shape[0]])
        for i in range(len(ps)):
            for j in range(len(ps)):
                if not i==j:
                    distances[i].append(distance(ps[i],ps[j]))
            distances[i].sort()
            di=np.mean(distances[i][0:3])*belta
            if ps[i].has_key('w') and ps[i].has_key('h'):
                       dy= np.zeros([img.shape[1],img.shape[0]])
                       x=(ps[i]['w']+ps[i]['x'])/2
                       y=(ps[i]['h']+ps[i]['y'])/2
                       dy[x][y]=1
                       density+=process(dy,di)
            elif ps[i].has_key('x') and ps[i].has_key('y'):
               while ps[i]['x']>density.shape[1]:
                     ps[i]['x']-=1
               dy= np.zeros([img.shape[1],img.shape[0]]);
               dy[ps[i]['x']][ps[i]['y']]=1
               density+=process(dy,di)
        s=sum(density)
        num=annotation['num']
        sub=num-s
        np.save('./ground_truth/'+str(annotation['id'])+'.npy',density)
        id[annotation['id']]=1
        if abs(sub)>1:
           print str(num)+' '+str(s)+' '+str(sub)+' '+str(annotation['id'])
run1=job_server.submit(main,(annotations,id),(sum,process,distance),())
run2=job_server.submit(main,(annotations,id),(sum,process,distance),())
run3=job_server.submit(main,(annotations,id),(sum,process,distance),())
run4=job_server.submit(main,(annotations,id),(sum,process,distance),())  
run2();run3();run4()

