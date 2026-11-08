"""Small arithmetic counterexample, not an empirical utility measurement."""
import json

def auc(labels, scores):
    pos=[s for y,s in zip(labels,scores) if y==1]
    neg=[s for y,s in zip(labels,scores) if y==0]
    return sum((p>n)+0.5*(p==n) for p in pos for n in neg)/(len(pos)*len(neg))
print(json.dumps({'seed_1_auc':auc([0,1],[.1,.2]),'seed_2_auc':auc([0,1],[.8,.9]),'pooled_raw_score_auc':auc([0,1,0,1],[.1,.2,.8,.9])},indent=2))
