import ast,pathlib,types,tempfile,os,json,glob
root=pathlib.Path.cwd()
source=(root/'code/evaluation/audit_dataset_completeness.py').read_text()
tree=ast.parse(source)
select=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Assign))]
ns={'os':os,'glob':glob,'json':json,'Optional':__import__('typing').Optional,'Dict':dict,'Any':object}
ns['np']=types.SimpleNamespace(asanyarray=lambda x:x,unique=lambda x:x)
ns['nib']=types.SimpleNamespace(load=lambda p:types.SimpleNamespace(dataobj=ns['labels']))
exec(compile(ast.Module(body=select,type_ignores=[]),'audit_case_ast','exec'),ns)
with tempfile.TemporaryDirectory() as d:
 pathlib.Path(d,'ct.nii.gz').touch(); pathlib.Path(d,'gi_mask.nii.gz').touch()
 for labels,meta in [([0,1,2,3,4],None),([0,1,2,4],{'schema':'canonical_5class','reviewed_classes':['stomach','duodenum','small_bowel','colon']}),([0,5],{'schema':'jhu','reviewed_classes':['colon']})]:
  mp=pathlib.Path(d,'case_metadata.json')
  if meta is not None:mp.write_text(json.dumps(meta))
  elif mp.exists():mp.unlink()
  ns['labels']=labels
  rec=ns['audit_case'](d)
  print(json.dumps({'labels':labels,'metadata':meta,'result':{k:rec[k] for k in ['schema','schema_verified','reviewed_classes','small_bowel_present','colon_present','is_complete_gi','status']}}))
for fn in ['train_mednext.py','train_swin_unetr.py','train_mednext_phase2.py','train_swin_unetr_phase2.py','run_scaling_sweep.py']:
 t=ast.parse((root/'code/training'/fn).read_text())
 strings=[n.value for n in ast.walk(t) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
 print(fn,'case_metadata.json tokens=',strings.count('case_metadata.json'),'reviewed_classes tokens=',strings.count('reviewed_classes'))
 if fn=='train_swin_unetr.py':
  for node in ast.walk(t):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='MarginalDiceCELoss':print('Swin marginal arguments AST',[(k.arg,type(k.value).__name__) for k in node.keywords])
try: print({1.0} * 1.2)
except Exception as e:print('Swin configured weight arithmetic:',type(e).__name__,str(e))
