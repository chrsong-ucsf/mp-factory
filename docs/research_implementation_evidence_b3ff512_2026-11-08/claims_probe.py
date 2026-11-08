"""Lightweight temporary probes. Run from mp-factory root; no training/model imports."""
import importlib.util, tempfile, contextlib, io, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

p9=load('code/evaluation/benchmark_p9_independent_study.py','p9')
with tempfile.TemporaryDirectory() as td:
    p=Path(td);gt=p/'gt';gt.mkdir();(gt/'not_real.nii.gz').touch()
    result=p9.run_benchmark(str(gt),str(p/'NOteacher'),str(p/'NOmajority'),str(p/'NOensemble'),str(p/'NOstudent'),str(p/'output'))
    print('Invalid GT and missing arms:',result,'files',list((p/'output').iterdir()))
    rows=[]
    for scan in range(5):
        for organ in ['stomach','duodenum','small_bowel','colon']:
            rows.extend([dict(patient_id='ONE',scan_id=str(scan),organ=organ,arm='student',dice=.8),dict(patient_id='ONE',scan_id=str(scan),organ=organ,arm='teacher',dice=.6)])
    print('One-patient inference:',p9.paired_cluster_bootstrap(pd.DataFrame(rows),'student','teacher',n_boot=20))
    a=np.zeros((64,64,64),bool);b=a.copy();a[10:40,10:40,10:40]=True;b[11:41,10:40,10:40]=True
    sa=a&~binary_erosion(a);sb=b&~binary_erosion(b)
    reference=np.percentile(np.r_[distance_transform_edt(~sa)[sb],distance_transform_edt(~sb)[sa]],95)
    print('HD95 implemented/surface:',p9.compute_hd95_numpy(a,b),reference)
    scaling=load('code/evaluation/compute_scaling_law.py','scaling')
    rows=[dict(n_train_scans=n,n_train_patients=[2,3,5,10][i],patient_id='ONE',mean_dice=.3+.1*i+.01*j) for i,n in enumerate([5,10,20,40]) for j in range(2)]
    pd.DataFrame(rows).to_csv(p/'scale.csv',index=False)
    with contextlib.redirect_stdout(io.StringIO()),warnings.catch_warnings():
        warnings.simplefilter('ignore');r=scaling.fit_scaling_models(str(p/'scale.csv'),n_boot=5)
    print('Scaling N/cluster mode/entities:',[x['N'] for x in r['data_points']],r.get('bootstrap_ci',{}).get('clustering_mode'),r.get('bootstrap_ci',{}).get('cluster_entity_count'))
