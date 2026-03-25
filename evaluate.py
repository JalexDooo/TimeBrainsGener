import os
import argparse
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.dataset import TimeBrainsDataset
from models.model import TimeBrainsModel


def save_nifti(tensor, filename, affine=np.eye(4)):
    img = nib.Nifti1Image(tensor.astype(np.float32), affine)
    nib.save(img, filename)


def save_2d_slice(slice_data, filename, cmap='gray'):
    plt.figure(figsize=(6, 6))
    plt.imshow(np.rot90(slice_data), cmap=cmap)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


def compute_ssim_3d(pred, gt):
    """Simple per-channel SSIM approximation using local statistics."""
    from skimage.metrics import structural_similarity
    ssim_vals = []
    for c in range(pred.shape[0]):
        data_range = gt[c].max() - gt[c].min()
        if data_range < 1e-8:
            continue
        s = structural_similarity(pred[c], gt[c], data_range=data_range)
        ssim_vals.append(s)
    return np.mean(ssim_vals) if ssim_vals else 0.0


def compute_psnr(pred, gt):
    mse = np.mean((pred - gt) ** 2)
    if mse < 1e-10:
        return 100.0
    data_range = gt.max() - gt.min()
    if data_range < 1e-8:
        return 0.0
    return 10 * np.log10((data_range ** 2) / mse)


def compute_dice_per_class(pred_seg, gt_seg, num_classes=4):
    """Compute Dice per foreground class."""
    dices = {}
    for c in range(1, num_classes):
        p = (pred_seg == c).astype(np.float32)
        g = (gt_seg == c).astype(np.float32)
        inter = (p * g).sum()
        union = p.sum() + g.sum()
        if union < 1e-8:
            dices[c] = float('nan')
        else:
            dices[c] = 2.0 * inter / union
    return dices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--data_dir', type=str,
                        default="/home/u6dm/jindong.u6dm/Code/dataset/clean_sailor_ebrains_pseud")
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to evaluate')
    args = parser.parse_args()
    ep = args.epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    dataset = TimeBrainsDataset(args.data_dir, is_train=False, return_full=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    vocab_size = max(10, len(dataset.treatment_vocab) + 5)
    model = TimeBrainsModel(in_channels=7, out_channels=3, num_classes=4,
                            base_feat=48, max_treatments=vocab_size).to(device)
    
    checkpoint_path = f"checkpoints/timebrainsgen_v15_epoch_{ep}.pth"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(f"WARNING: {checkpoint_path} not found. Using random weights.")

    model.eval()
    os.makedirs("predictions", exist_ok=True)
    
    all_ssim, all_psnr, all_dice = [], [], []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= args.num_samples:
                break
                
            img_t = batch['image_t'].to(device)
            seg_t = batch['seg_t'].to(device)
            img_next_gt = batch['image_next'].to(device)
            seg_next_gt = batch['seg_next'].to(device)
            dt = batch['dt'].unsqueeze(-1).to(device) / 365.0
            treatment = batch['treatment'].to(device)
            
            seg_t_onehot = F.one_hot(seg_t.long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
            model_input_t = torch.cat([img_t, seg_t_onehot], dim=1)
            
            delta_pred, seg_next_pred_logits, _, _ = model(model_input_t, dt, treatment)
            seg_next_pred_class = torch.argmax(F.softmax(seg_next_pred_logits, dim=1), dim=1)
            
            # --- Reconstruct Full Volume from ROI ---
            img_t_full = batch['image_t_full'].to(device)
            seg_t_full = batch['seg_t_full'].to(device)
            img_next_gt_full = batch['image_next_full'].to(device)
            seg_next_gt_full = batch['seg_next_full'].to(device)
            
            bbox = batch['bbox'][0] # batch_size is 1
            d_min, d_max, h_min, h_max, w_min, w_max = bbox
            
            img_next_pred_full = img_t_full.clone()
            img_next_pred_full[:, :, d_min:d_max, h_min:h_max, w_min:w_max] = img_t + delta_pred
            
            seg_next_pred_full = torch.zeros_like(seg_t_full)
            seg_next_pred_full[:, d_min:d_max, h_min:h_max, w_min:w_max] = seg_next_pred_class
            
            delta_pred_full = torch.zeros_like(img_t_full)
            delta_pred_full[:, :, d_min:d_max, h_min:h_max, w_min:w_max] = delta_pred
            
            delta_gt_full = img_next_gt_full - img_t_full
            
            # To numpy
            img_t_np = img_t_full.cpu().numpy()[0]
            seg_t_np = seg_t_full.cpu().numpy()[0]
            img_next_np = img_next_gt_full.cpu().numpy()[0]
            seg_next_np = seg_next_gt_full.cpu().numpy()[0]
            img_pred_np = img_next_pred_full.cpu().numpy()[0]
            seg_pred_np = seg_next_pred_full.cpu().numpy()[0]
            delta_pred_np = delta_pred_full.cpu().numpy()[0]
            delta_gt_np = delta_gt_full.cpu().numpy()[0]
            
            # --- Quantitative metrics ---
            ssim_val = compute_ssim_3d(img_pred_np, img_next_np)
            psnr_val = compute_psnr(img_pred_np, img_next_np)
            dice_vals = compute_dice_per_class(seg_pred_np, seg_next_np)
            
            # Mean dice over existing foreground classes
            valid_dices = [v for v in dice_vals.values() if not np.isnan(v)]
            mean_dice = np.mean(valid_dices) if valid_dices else 0.0
            
            all_ssim.append(ssim_val)
            all_psnr.append(psnr_val)
            all_dice.append(mean_dice)
            
            # Delta statistics
            print(f"Sample {batch_idx}: SSIM={ssim_val:.4f} PSNR={psnr_val:.2f} "
                  f"Dice(fg)={mean_dice:.4f} "
                  f"[c1={dice_vals.get(1, float('nan')):.3f} "
                  f"c2={dice_vals.get(2, float('nan')):.3f} "
                  f"c3={dice_vals.get(3, float('nan')):.3f}] "
                  f"delta_pred_std={delta_pred_np.std():.4f} "
                  f"delta_gt_std={delta_gt_np.std():.4f}")

            # --- Save outputs ---
            def remap_brats(arr):
                res = arr.copy()
                res[res == 3] = 4
                return res

            # Save NIfTI volumes for every sample
            save_nifti(img_t_np[0], f"predictions/epoch{ep}_sample_{batch_idx}_t_T1.nii.gz")
            save_nifti(remap_brats(seg_t_np), f"predictions/epoch{ep}_sample_{batch_idx}_t_seg.nii.gz")
            save_nifti(img_next_np[0], f"predictions/epoch{ep}_sample_{batch_idx}_next_gt_T1.nii.gz")
            save_nifti(remap_brats(seg_next_np), f"predictions/epoch{ep}_sample_{batch_idx}_next_gt_seg.nii.gz")
            save_nifti(img_pred_np[0], f"predictions/epoch{ep}_sample_{batch_idx}_next_pred_T1.nii.gz")
            save_nifti(remap_brats(seg_pred_np), f"predictions/epoch{ep}_sample_{batch_idx}_next_pred_seg.nii.gz")
            save_nifti(delta_pred_np[0], f"predictions/epoch{ep}_sample_{batch_idx}_delta_pred.nii.gz")
            save_nifti(delta_gt_np[0], f"predictions/epoch{ep}_sample_{batch_idx}_delta_gt.nii.gz")
            
            # 2D slices
            tumor_pixels_per_z = np.sum(seg_next_np > 0, axis=(0, 1))
            best_z = np.argmax(tumor_pixels_per_z)
            if tumor_pixels_per_z[best_z] == 0:
                best_z = seg_next_np.shape[2] // 2

            save_2d_slice(img_t_np[0, :, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_t_T1.png")
            save_2d_slice(seg_t_np[:, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_t_seg.png", cmap='magma')
            save_2d_slice(img_next_np[0, :, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_next_gt_T1.png")
            save_2d_slice(img_pred_np[0, :, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_next_pred_T1.png")
            save_2d_slice(seg_next_np[:, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_next_gt_seg.png", cmap='magma')
            save_2d_slice(seg_pred_np[:, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_next_pred_seg.png", cmap='magma')
            
            # Delta maps: use RdBu_r colormap (blue=decrease, red=increase, white=no change)
            vmax = max(abs(delta_gt_np[0, :, :, best_z]).max(), abs(delta_pred_np[0, :, :, best_z]).max(), 0.01)
            save_2d_slice(delta_gt_np[0, :, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_delta_gt.png", cmap='RdBu_r')
            save_2d_slice(delta_pred_np[0, :, :, best_z],
                          f"predictions/epoch{ep}_2d_sample_{batch_idx}_delta_pred.png", cmap='RdBu_r')

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"Epoch {ep} Evaluation Summary ({len(all_ssim)} samples)")
    print(f"  SSIM:       {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    print(f"  PSNR:       {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
    print(f"  Dice (fg):  {np.mean(all_dice):.4f} ± {np.std(all_dice):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
