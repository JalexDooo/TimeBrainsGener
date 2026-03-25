import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

try:
    import monai.transforms as mt
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not detected. Data augmentation disabled.")

# Hardcoded vocab to guarantee consistency across all DDP ranks
TREATMENT_VOCAB = {"unknown": 0, "CRT": 1, "TMZ": 2, "no": 3}

class TimeBrainsDataset(Dataset):
    def __init__(self, root_dir, is_train=False, patient_ids=None, return_full=False):
        """
        Args:
            root_dir: path to the dataset root
            is_train: if True, apply data augmentation
            patient_ids: optional list of patient folder names
            return_full: if True, return original uncropped images for evaluation reconstruction
        """
        self.root_dir = Path(root_dir)
        self.is_train = is_train
        self.return_full = return_full
        self.pairs = []
        self.treatment_vocab = TREATMENT_VOCAB
        
        if MONAI_AVAILABLE and self.is_train:
            self.augs = mt.Compose([
                # --- Spatial augmentations (synchronized across all 4 volumes) ---
                mt.RandFlipd(keys=["image_t", "seg_t", "image_next", "seg_next"], spatial_axis=[0], prob=0.5),
                mt.RandFlipd(keys=["image_t", "seg_t", "image_next", "seg_next"], spatial_axis=[1], prob=0.5),
                mt.RandFlipd(keys=["image_t", "seg_t", "image_next", "seg_next"], spatial_axis=[2], prob=0.5),
                mt.RandAffined(
                    keys=["image_t", "seg_t", "image_next", "seg_next"],
                    prob=0.4,
                    rotate_range=(0.15, 0.15, 0.15),
                    translate_range=(8, 8, 8),
                    scale_range=(0.1, 0.1, 0.1),
                    mode=("bilinear", "nearest", "bilinear", "nearest")
                ),
                # --- Intensity augmentations (images only, not seg masks) ---
                mt.RandGaussianNoised(keys=["image_t", "image_next"], prob=0.3, mean=0.0, std=0.05),
                mt.RandScaleIntensityd(keys=["image_t", "image_next"], factors=0.15, prob=0.3),
                mt.RandShiftIntensityd(keys=["image_t", "image_next"], offsets=0.1, prob=0.3),
            ])
        
        all_subs = sorted([d.name for d in self.root_dir.glob("sub-*") if d.is_dir()])
        if patient_ids is not None:
            all_subs = [s for s in all_subs if s in patient_ids]
        
        for sub_id in all_subs:
            sub_dir = self.root_dir / sub_id
            intervals = []
            interval_file = sub_dir / "intervals-days.txt"
            if interval_file.exists():
                with open(interval_file, 'r') as f:
                    intervals = [float(line.strip()) for line in f if line.strip()]
            
            sessions = sorted([d.name for d in sub_dir.glob("ses-*") if d.is_dir()])
            if len(sessions) < 2:
                continue
                
            for i in range(len(sessions) - 1):
                ses_t = sessions[i]
                ses_next = sessions[i + 1]
                dt = intervals[i] if i < len(intervals) else 30.0
                t_path = sub_dir / ses_t
                next_path = sub_dir / ses_next
                
                if self._has_mods(t_path) and self._has_mods(next_path):
                    treatment = "unknown"
                    trt_file = t_path / "treatment.txt"
                    if trt_file.exists():
                        with open(trt_file, 'r') as f:
                            treatment = f.read().strip()
                    
                    # Map unseen treatments to "unknown" instead of dynamically expanding vocab
                    if treatment not in self.treatment_vocab:
                        treatment = "unknown"
                    
                    self.pairs.append({
                        "sub": sub_id,
                        "t_dir": t_path,
                        "next_dir": next_path,
                        "dt": dt,
                        "treatment": treatment
                    })

    @staticmethod
    def _has_mods(p):
        return (p / "T1-icor-zscore.nii.gz").exists() and \
               (p / "T1c-icor-zscore.nii.gz").exists() and \
               (p / "Flair-icor-zscore.nii.gz").exists() and \
               (p / "Segmentation-ONCO.nii.gz").exists()

    @staticmethod
    def get_all_patient_ids(root_dir):
        """Return sorted list of all patient folder names."""
        return sorted([d.name for d in Path(root_dir).glob("sub-*") if d.is_dir()])
                    
    def load_image(self, ses_path):
        vols = []
        for mod in ["T1-icor-zscore.nii.gz", "T1c-icor-zscore.nii.gz", "Flair-icor-zscore.nii.gz"]:
            img = nib.load(ses_path / mod).get_fdata()
            vols.append(np.nan_to_num(img))
        stack = np.stack(vols, axis=0).astype(np.float32)
        return torch.from_numpy(stack)
        
    def load_seg(self, ses_path):
        img = nib.load(ses_path / "Segmentation-ONCO.nii.gz").get_fdata()
        img = np.nan_to_num(img)
        img[img == 4] = 3
        return torch.from_numpy(img).float()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        sample = {
            "image_t": self.load_image(pair["t_dir"]),
            "seg_t": self.load_seg(pair["t_dir"]).unsqueeze(0),
            "image_next": self.load_image(pair["next_dir"]),
            "seg_next": self.load_seg(pair["next_dir"]).unsqueeze(0),
        }
        # --- ROI Extraction (128x128x128) ---
        roi_size = 128
        img_t, seg_t, img_next, seg_next = sample["image_t"], sample["seg_t"], sample["image_next"], sample["seg_next"]
        tumor_mask = seg_t[0] > 0
        coords = torch.nonzero(tumor_mask)
        
        if len(coords) == 0:
            D_dim, H_dim, W_dim = seg_t.shape[1:]
            center = torch.tensor([D_dim // 2, H_dim // 2, W_dim // 2])
        else:
            min_coords = coords.min(dim=0)[0]
            max_coords = coords.max(dim=0)[0]
            center = (min_coords + max_coords) // 2

        D_dim, H_dim, W_dim = seg_t.shape[1:]
        half_size = roi_size // 2
        d_min = max(0, int(center[0]) - half_size)
        d_max = min(D_dim, int(center[0]) + half_size)
        h_min = max(0, int(center[1]) - half_size)
        h_max = min(H_dim, int(center[1]) + half_size)
        w_min = max(0, int(center[2]) - half_size)
        w_max = min(W_dim, int(center[2]) + half_size)
        
        # Adjust boundaries if they hit the edge to guarantee 128x128x128
        if d_max - d_min < roi_size:
            if d_min == 0: d_max = min(D_dim, roi_size)
            else: d_min = max(0, D_dim - roi_size)
        if h_max - h_min < roi_size:
            if h_min == 0: h_max = min(H_dim, roi_size)
            else: h_min = max(0, H_dim - roi_size)
        if w_max - w_min < roi_size:
            if w_min == 0: w_max = min(W_dim, roi_size)
            else: w_min = max(0, W_dim - roi_size)
            
        sample = {
            "image_t": img_t[:, d_min:d_max, h_min:h_max, w_min:w_max],
            "seg_t": seg_t[:, d_min:d_max, h_min:h_max, w_min:w_max],
            "image_next": img_next[:, d_min:d_max, h_min:h_max, w_min:w_max],
            "seg_next": seg_next[:, d_min:d_max, h_min:h_max, w_min:w_max],
        }
        
        if MONAI_AVAILABLE and self.is_train:
            sample = self.augs(sample)
            
        seg_t_out = sample["seg_t"].squeeze(0).long()
        seg_next_out = sample["seg_next"].squeeze(0).long()
        
        trt_idx = self.treatment_vocab[pair["treatment"]]
            
        ret = {
            "image_t": sample["image_t"].clone(),
            "seg_t": seg_t_out.clone(),
            "image_next": sample["image_next"].clone(),
            "seg_next": seg_next_out.clone(),
            "dt": torch.tensor(pair["dt"], dtype=torch.float32),
            "treatment": torch.tensor(trt_idx, dtype=torch.long),
            "bbox": torch.tensor([d_min, d_max, h_min, h_max, w_min, w_max], dtype=torch.long)
        }
        
        if self.return_full:
            ret["image_t_full"] = img_t.clone()
            ret["seg_t_full"] = seg_t.squeeze(0).long().clone()
            ret["image_next_full"] = img_next.clone()
            ret["seg_next_full"] = seg_next.squeeze(0).long().clone()
            
        return ret
