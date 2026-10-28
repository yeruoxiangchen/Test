"""Export comparison figures only from complete aggregate reports."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/palr_matplotlib')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .diagnostic_suite import OUT


def main():
    reports={k:json.loads((OUT/k/'aggregate.json').read_text()) for k in ('pilot32','dev64','strong16','mesh')}
    if not all(v['complete_pairing'] for v in reports.values()):raise ValueError('partial results cannot be plotted')
    names={'parent':'Parent SS30K','v1':'V1 SS 2000','raw':'V2 SS 2000 (raw weights)','bias_only':'V2: biases only','layer_view_roll':'V2: mismatched layers',
           'baseline500':'EMA init + joint','raw_joint':'Raw init + joint','raw_frozen':'Raw init + frozen',
           'raw_frozen_v1recipe':'Frozen + V1 recipe','raw_frozen_v1recipe_tol004':'V1 recipe + tol .04'}
    colors=['#9aa0a6','#5886c4','#e5a041','#2d9b88','#a56cb8','#ce6d70']
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,key,title in zip(axes.flat,('pilot32','dev64','strong16','mesh'),
            ('500-step SS; 32 training objects','64 unseen objects',
             '16 unseen objects; >=4 views with <50% visibility','Frozen SLat mesh; 32 training objects')):
        metrics=reports[key]['groups']['all']['metrics'];modes=list(metrics)
        metric='chamfer_l1' if key=='mesh' else 'iou'
        values=[metrics[m][metric] for m in modes]
        ax.barh(np.arange(len(modes)),values,color=colors[:len(modes)])
        ax.set_yticks(np.arange(len(modes)),[names[m] for m in modes]);ax.invert_yaxis()
        ax.set_xlim(0,max(values)*1.22);ax.set_title(title)
        ax.set_xlabel('Native-frame Chamfer L1 (lower is better)' if key=='mesh' else 'SS IoU (higher is better)')
        for y,v in enumerate(values):ax.text(v,y,f'  {v:.4f}',va='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Paired inputs and inference seeds 42 / 43; one training seed',fontsize=15)
    fig.supxlabel('Train32: clean + first occluded input. Stress: synthetic, original cameras. Mesh reference: official VAE proxy.',fontsize=9)
    fig.savefig(OUT/'diagnostic_results.png',dpi=160);plt.close(fig)
    effects=[]
    for title,label,key in [
        ('Raw vs EMA initialization (500 steps)','pilot32','raw_joint_minus_baseline500'),
        ('Freeze vs joint geometry (500 steps)','pilot32','raw_frozen_minus_raw_joint'),
        ('V1 recipe vs current recipe (500 steps)','pilot32','raw_frozen_v1recipe_minus_raw_frozen'),
        ('Tolerance .04 vs .02 (500 steps)','pilot32','raw_frozen_v1recipe_tol004_minus_raw_frozen_v1recipe'),
        ('Unseen objects: V2 raw vs parent','dev64','raw_minus_parent'),
        ('Unseen objects: V2 raw vs V1','dev64','raw_minus_v1'),
        ('Strong occlusion: V2 raw vs parent','strong16','raw_minus_parent'),
        ('Strong occlusion: V2 raw vs V1','strong16','raw_minus_v1')]:
        effects.append((title,reports[label]['groups']['all']['paired'][key]))
    if 'raw_minus_bias_only' in reports['dev64']['groups']['all']['paired']:
        effects.append(('Unseen objects: full geometry vs biases only',reports['dev64']['groups']['all']['paired']['raw_minus_bias_only']))
    if 'raw_minus_layer_view_roll' in reports['dev64']['groups']['all']['paired']:
        effects.append(('Unseen objects: correct vs misaligned layer maps',reports['dev64']['groups']['all']['paired']['raw_minus_layer_view_roll']))
    fig,ax=plt.subplots(figsize=(12,6),layout='constrained')
    for i,(title,r) in enumerate(effects):
        center=r['mean_improvement']*100;lo,hi=np.array(r['object_bootstrap_ci95'])*100
        ax.plot([lo,hi],[i,i],color='#376f91',linewidth=2)
        ax.scatter([center],[i],color='#c45b3e',zorder=3)
    ax.set_yticks(np.arange(len(effects)),[r[0] for r in effects]);ax.invert_yaxis()
    ax.axvline(0,color='gray',linestyle='--');ax.set_xlabel('SS IoU improvement (percentage points), object bootstrap 95% CI')
    ax.set_title('Paired diagnostic effects; positive means the first condition performs better')
    fig.supxlabel('Exploratory intervals over objects, conditional on one training run; no multiple-comparison correction.',fontsize=9)
    ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'paired_effects.png',dpi=160);plt.close(fig)
    print(OUT/'diagnostic_results.png');print(OUT/'paired_effects.png')


if __name__=='__main__':main()
