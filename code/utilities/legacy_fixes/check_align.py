import nibabel as nib
import numpy as np

sub = 'BDMAP_00242113'
base = f'/mnt/scratch/user/chrsong/mp-factory/CancerVerse_dbox/{sub}'

ct = nib.load(f'{base}/ct.nii.gz')
gi = nib.load(f'{base}/gi_mask_temp.nii.gz')
intestine = nib.load(f'{base}/segmentations/intestine.nii.gz')
consensus = nib.load(f'/mnt/scratch/user/chrsong/mp-factory/results/ensemble_out/{sub}_consensus.nii.gz')

print("CT:", ct.shape)
print("GI:", gi.shape)
print("Intestine:", intestine.shape)
print("Consensus:", consensus.shape)
