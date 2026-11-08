"""Synthetic temporary controls. Run from mp-factory root. No source edits or model imports.
First run is UNMODIFIED. A local in-memory missing-helper alias then permits downstream
geometry/empty-prediction diagnostics; these alias-isolated cases do not certify normal execution.
"""
import importlib.util, tempfile, contextlib, io, json, warnings
from pathlib import Path
import pandas as pd
import numpy as np
import nibabel as nib

def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

m=load('code/evaluation/benchmark_p9_independent_study.py','p9')
with tempfile.TemporaryDirectory() as td:
    p=Path(td);gt=p/'gt';gt.mkdir();arms=[p/f'arm{i}' for i in range(4)]
    for d in arms:d.mkdir()
    a=np.zeros((8,8,8),np.uint8);a[2:6,2:6,2:6]=1;sid='BDMAP_00242113'
    nib.save(nib.Nifti1Image(a,np.eye(4)),gt/f'{sid}.nii.gz')
    for d in arms:nib.save(nib.Nifti1Image(a,np.eye(4)),d/f'{sid}_seg.nii.gz')
    r=m.run_benchmark(str(gt),*[str(d) for d in arms],str(p/'original'))
    print('UNMODIFIED:',r['status'],pd.read_csv(p/'original/p9_matched_predictions.csv').status.unique().tolist())
    m.compute_betti0_error=m.compute_betti0_numpy  # In-memory diagnostic alias ONLY
    aff=np.eye(4);aff[0,3]=1000
    for d in arms:nib.save(nib.Nifti1Image(a,aff),d/f'{sid}_seg.nii.gz')
    m.run_benchmark(str(gt),*[str(d) for d in arms],str(p/'displaced'))
    print('ALIAS-ISOLATED displaced:',pd.read_csv(p/'displaced/p9_matched_predictions.csv').to_dict('records'))
    for d in arms:nib.save(nib.Nifti1Image(np.zeros_like(a),np.eye(4)),d/f'{sid}_seg.nii.gz')
    m.run_benchmark(str(gt),*[str(d) for d in arms],str(p/'empty'))
    print('ALIAS-ISOLATED empty:',pd.read_csv(p/'empty/p9_matched_predictions.csv').to_dict('records'))
    sc=load('code/evaluation/compute_scaling_law.py','sc');rows=[]
    for i,n in enumerate([5,10,20,40]):
        for j in range(2):rows.append(dict(train_scan_count=n,train_patient_count=[2,3,5,10][i],scan_id=f'scan{i}_{j}',mean_dice=.3+.1*i+.01*j))
    pd.DataFrame(rows).to_csv(p/'s.csv',index=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');r=sc.fit_scaling_models(str(p/'s.csv'),n_boot=5,x_axis='annotation_budget')
    print('Missing annotation budget fallback:',[z['N'] for z in r['data_points']],r['patient_cluster_column'],r.get('bootstrap_ci',{}).get('clustering_mode'))
