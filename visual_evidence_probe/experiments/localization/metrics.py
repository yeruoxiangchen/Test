"""Tie-correct ranking metrics, reporting undefined/low-coverage cases explicitly."""
import numpy as np


def binary_ranking(labels,scores):
    y=np.asarray(labels,bool).ravel();s=np.asarray(scores,float).ravel()
    if len(y)!=len(s) or not np.isfinite(s).all():raise ValueError('invalid score arrays')
    n=len(y);positive=int(y.sum());negative=n-positive
    base=dict(count=n,positive=positive,negative=negative,prevalence=positive/n if n else None)
    if not positive or not negative:
        return dict(**base,defined=False,ap=None,auroc=None,ap_lift=None,precision_top20=None,recall_top20=None)
    order=np.argsort(-s,kind='stable');sorted_s=s[order];sorted_y=y[order]
    ends=np.r_[np.flatnonzero(sorted_s[:-1]!=sorted_s[1:]),n-1]
    tp=np.cumsum(sorted_y)[ends].astype(float);fp=(ends+1)-tp
    recall=tp/positive;precision=tp/(ends+1)
    ap=float(np.sum(np.diff(np.r_[0.,recall])*precision))
    tpr=np.r_[0.,recall];fpr=np.r_[0.,fp/negative]
    auroc=float(np.trapz(tpr,fpr))
    # Tie-neutral expected precision for exactly 20% of pixels.
    take=max(1,int(np.ceil(.2*n)));threshold=sorted_s[take-1]
    high=s>threshold;tie=s==threshold
    selected_positive=float(y[high].sum())+(take-high.sum())*float(y[tie].mean())
    return dict(**base,defined=True,ap=ap,auroc=auroc,ap_lift=ap/(positive/n),
                precision_top20=selected_positive/take,recall_top20=selected_positive/positive)


def evaluate(maps,pred_depth,gt_depth,gt_mask,threshold):
    pm=maps['pred_mask'];error=np.abs(pred_depth-gt_depth)
    labels=(pm ^ gt_mask)|((pm & gt_mask)&(error>threshold))
    common=maps['common']
    interior=maps['interior']&maps['anchor_valid']
    union=(pm|maps['observed_mask'])&maps['anchor_valid']
    methods=('silhouette','boundary','direct_rgb','direct_dino','cross_rgb','cross_dino','shuffled_dino','fused_dino')
    from scipy.stats import rankdata
    ranked_depth=rankdata(error[common]) if common.any() else np.empty(0)
    continuous={}
    for key in methods:
        ranked_score=rankdata(maps[key][common]) if common.any() else np.empty(0)
        valid=len(ranked_depth)>1 and np.std(ranked_depth)>0 and np.std(ranked_score)>0
        continuous[key]=float(np.corrcoef(ranked_score,ranked_depth)[0,1]) if valid else None
    return dict(threshold=threshold,interior_pixels=int(interior.sum()),common_pixels=int(common.sum()),
        common_coverage=float(common.sum()/max(interior.sum(),1)),
        interior_error_rate=float(labels[interior].mean()) if interior.any() else None,
        common_error_rate=float(labels[common].mean()) if common.any() else None,
        common={key:binary_ranking(labels[common],maps[key][common]) for key in methods},
        common_depth_error_spearman=continuous,
        full={key:binary_ranking(labels[union],maps[key][union]) for key in ('silhouette','boundary','direct_rgb','direct_dino','fused_full')})
