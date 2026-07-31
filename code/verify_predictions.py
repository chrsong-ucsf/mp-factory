import os
import sys
import glob
import numpy as np
import nibabel as nib

ORGAN_MAP = {
    0: 'Background',
    1: 'Stomach',
    2: 'Duodenum',
    3: 'Small Bowel / Intestine',
    4: 'Colon'
}

def verify_predictions(pred_dir, max_files=10):
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    total_files = len(pred_files)
    
    print("==========================================================")
    print("        Swin-UNETR Segmentation Verification Script       ")
    print("==========================================================")
    print(f"Prediction Directory : {pred_dir}")
    print(f"Total Predicted Files: {total_files}")
    print("==========================================================\n")
    
    if total_files == 0:
        print("No prediction NIfTI (.nii.gz) files found in destination directory yet.")
        return

    sample_files = pred_files[:max_files]
    
    for idx, fpath in enumerate(sample_files, 1):
        filename = os.path.basename(fpath)
        print(f"[{idx}/{len(sample_files)}] Inspecting: {filename}")
        try:
            nii = nib.load(fpath)
            data = nii.get_fdata().astype(np.uint8)
            print(f"  --> 3D Shape: {data.shape} | Voxel Data Type: {data.dtype}")
            
            unique_labels, counts = np.unique(data, return_counts=True)
            print("  --> Organ Voxel Counts:")
            for lbl, count in zip(unique_labels, counts):
                organ_name = ORGAN_MAP.get(int(lbl), f"Unknown ({lbl})")
                pct = (count / data.size) * 100
                print(f"        Label {int(lbl)} ({organ_name}): {count:,} voxels ({pct:.2f}%)")
            print("")
        except Exception as e:
            print(f"  [ERROR] Failed to read NIfTI file {filename}: {e}\n")

if __name__ == "__main__":
    pred_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/scratch/user/chrsong/mp-factory/results/swin_unetr_predictions"
    verify_predictions(pred_dir)
