import ast,pathlib,os,glob,json,tempfile,types,math
root=pathlib.Path.cwd()
for fname in ['train_mednext.py','train_swin_unetr.py']:
 tree=ast.parse((root/'code/training'/fname).read_text())
 fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='discover_dataset')
 ns={'os':os,'glob':glob,'json':json,'ORGAN_ALIASES':{1:['stomach'],2:['duodenum'],3:['small_bowel'],4:['colon']}}
 exec(compile(ast.Module(body=[fn],type_ignores=[]),fname,'exec'),ns)
 with tempfile.TemporaryDirectory() as d:
  case=pathlib.Path(d,'case');case.mkdir();(case/'ct.nii.gz').touch();(case/'gi_mask.nii.gz').touch()
  for metadata in [{'schema':'jhu','reviewed_classes':['stomach'],'human_reviewed':False},{'schema':'canonical_5class','per_organ_provenance':{}}]:
   (case/'case_metadata.json').write_text(json.dumps(metadata))
   rec=ns['discover_dataset'](d)[0]
   print(fname,'metadata=',metadata,'reviewed=',rec.get('reviewed_classes'),'schema=',rec.get('schema'))
for fname in ['train_mednext_phase2.py','train_swin_unetr_phase2.py','run_scaling_sweep.py']:
 t=ast.parse((root/'code/training'/fname).read_text());strings=[n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
 print(fname,'metadata file tokens',strings.count('case_metadata.json'),'reviewed_classes tokens',strings.count('reviewed_classes'))
labels=[0,0,1];weights=[1,1,1];support=[int(m>0) for m in labels];den=sum(w*s for w,s in zip(weights,support));votes=[sum(w*s for w,s,m in zip(weights,support,labels) if m==c)/den for c in range(2)]
print('Actual hard-mask fallback equal-weight BG,BG,organ -> support',support,'consensus probabilities',votes,'winner',votes.index(max(votes)))
print('Current mean for supported entropy0.2 plus unsupportedNaN:',sum([0.2,float('nan')])/2)
order=[2,0,1,3,4];print('Non-self-inverse channel order',order,'current reordered channel labels',[order[i] for i in order],'expected',[0,1,2,3,4])
