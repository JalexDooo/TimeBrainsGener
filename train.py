import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from accelerate.utils import set_seed
import argparse

from datasets.dataset import TimeBrainsDataset
from models.model import TimeBrainsModel, Discriminator

class ForegroundDiceLoss(nn.Module):
    """Dice loss computed only on foreground classes (excludes class 0 = background)."""
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)
        true_1hot = F.one_hot(true.long(), num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        
        # Only compute dice on classes 1, 2, 3 (foreground)
        probs_fg = probs[:, 1:, ...]
        true_fg = true_1hot[:, 1:, ...]
        
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs_fg * true_fg, dims)
        cardinality = torch.sum(probs_fg + true_fg, dims)
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class segmentation.
    Focuses training on hard examples and rare classes (especially class 3, GD-enhanced).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights  # Optional per-class weights

    def forward(self, logits, targets):
        # logits: (B, C, D, H, W), targets: (B, D, H, W)
        B, C, D, H, W = logits.shape
        log_probs = F.log_softmax(logits, dim=1)   # (B, C, D, H, W)
        probs = torch.exp(log_probs)
        
        targets_1hot = F.one_hot(targets.long(), num_classes=C).permute(0, 4, 1, 2, 3).float()
        p_t = (probs * targets_1hot).sum(dim=1)     # (B, D, H, W)
        focal_weight = (1 - p_t) ** self.gamma
        
        nll = -(log_probs * targets_1hot).sum(dim=1) # per-voxel CE: (B, D, H, W)
        
        if self.class_weights is not None:
            w = self.class_weights.to(logits.device)
            per_class_w = (w.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4) * targets_1hot).sum(dim=1)
            nll = nll * per_class_w
        
        return (focal_weight * nll).mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_dir', type=str,
                        default="/home/u6dm/jindong.u6dm/Code/dataset/clean_sailor_ebrains_pseud")
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint directory to resume from')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of patients for validation')
    args = parser.parse_args()
    
    accelerator = Accelerator(mixed_precision='bf16')
    set_seed(42)
    
    accelerator.print(f"Processes: {accelerator.num_processes}, Mixed precision: {accelerator.mixed_precision}")

    # --- Patient-level train/val split ---
    all_patients = TimeBrainsDataset.get_all_patient_ids(args.data_dir)
    n_val = max(1, int(len(all_patients) * args.val_ratio))
    val_patients = all_patients[-n_val:]
    train_patients = all_patients[:-n_val]
    accelerator.print(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
    
    train_dataset = TimeBrainsDataset(args.data_dir, is_train=True, patient_ids=train_patients)
    val_dataset = TimeBrainsDataset(args.data_dir, is_train=False, patient_ids=val_patients)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    accelerator.print(f"Train pairs: {len(train_dataset)}, Val pairs: {len(val_dataset)}")

    # --- Model init ---
    vocab_size = max(10, len(train_dataset.treatment_vocab) + 5)
    model = TimeBrainsModel(in_channels=7, out_channels=3, num_classes=4,
                            base_feat=48, max_treatments=vocab_size)
    discriminator = Discriminator(in_channels=7)  # 3 MRI + 4 one-hot seg
    
    opt_g = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay=1e-5)
    
    scheduler_g = CosineAnnealingLR(opt_g, T_max=args.epochs, eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(opt_d, T_max=args.epochs, eta_min=1e-6)
    
    # Class weights: amplify rare classes (2=necrosis, 3=GD-enhanced are smallest)
    seg_class_weights = torch.tensor([0.5, 1.0, 2.0, 4.0])  # BG, TC, WT, GD
    criterion_dice = ForegroundDiceLoss()
    criterion_focal = FocalLoss(gamma=2.0, class_weights=seg_class_weights)
    criterion_bce = nn.BCEWithLogitsLoss()

    model, discriminator, opt_g, opt_d, train_loader, val_loader = accelerator.prepare(
        model, discriminator, opt_g, opt_d, train_loader, val_loader
    )
    
    # Schedulers should NOT be prepared by accelerate
    start_epoch = 0

    # --- Resume from checkpoint ---
    if args.resume and os.path.isdir(args.resume):
        ckpt_path = os.path.join(args.resume, "training_state.pth")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=accelerator.device)
            accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
            accelerator.unwrap_model(discriminator).load_state_dict(ckpt['discriminator'])
            opt_g.load_state_dict(ckpt['opt_g'])
            opt_d.load_state_dict(ckpt['opt_d'])
            scheduler_g.load_state_dict(ckpt['scheduler_g'])
            scheduler_d.load_state_dict(ckpt['scheduler_d'])
            start_epoch = ckpt['epoch']
            accelerator.print(f"Resumed from epoch {start_epoch}")

    epochs = args.epochs
    accelerator.print(f"Training for epochs {start_epoch+1} to {epochs}...")
    
    for epoch in range(start_epoch, epochs):
        # ==================== TRAIN ====================
        model.train()
        discriminator.train()
        epoch_g_loss = 0.0
        epoch_seg_loss = 0.0
        epoch_img_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            img_t = batch['image_t']
            seg_t = batch['seg_t']
            img_next_gt = batch['image_next']
            seg_next_gt = batch['seg_next']
            dt = batch['dt'].unsqueeze(-1) / 365.0
            treatment = batch['treatment']
            
            seg_t_onehot = F.one_hot(seg_t.long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
            model_input_t = torch.cat([img_t, seg_t_onehot], dim=1)
            
            # --- Discriminator (after warmup) ---
            if epoch >= 10:
                opt_d.zero_grad()
                model.eval()
                with torch.no_grad():
                    delta_pred, seg_pred_logits, _, _ = model(model_input_t, dt, treatment)
                    img_next_pred = img_t + delta_pred
                    seg_pred_onehot = F.one_hot(torch.argmax(F.softmax(seg_pred_logits, dim=1), dim=1), num_classes=4).permute(0, 4, 1, 2, 3).float()
                model.train()
                
                # Discriminator sees MRI + seg together
                seg_next_gt_onehot = F.one_hot(seg_next_gt.long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
                real_input = torch.cat([img_next_gt, seg_next_gt_onehot], dim=1)  # (B, 7, D, H, W)
                fake_input = torch.cat([img_next_pred.detach(), seg_pred_onehot.detach()], dim=1)
                
                combined = torch.cat([real_input, fake_input], dim=0)
                combined_logits = discriminator(combined)
                real_logits, fake_logits = combined_logits.chunk(2, dim=0)
                
                d_loss = (criterion_bce(real_logits, torch.ones_like(real_logits)) +
                          criterion_bce(fake_logits, torch.zeros_like(fake_logits))) / 2
                
                accelerator.backward(d_loss)
                accelerator.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                opt_d.step()
            
            # --- Generator ---
            opt_g.zero_grad()
            discriminator.eval()
            for p in discriminator.parameters():
                p.requires_grad = False
                
            delta_pred, seg_next_pred, z_t, z_next_pred = model(model_input_t, dt, treatment)
            img_next_pred = img_t + delta_pred  # Residual reconstruction
            
            # Adversarial: generator tries to fool discriminator with MRI + seg
            if epoch >= 10:
                seg_pred_onehot_g = F.one_hot(torch.argmax(F.softmax(seg_next_pred, dim=1), dim=1), num_classes=4).permute(0, 4, 1, 2, 3).float()
                fake_input_g = torch.cat([img_next_pred, seg_pred_onehot_g], dim=1)
                fake_logits_for_g = discriminator(fake_input_g)
                g_adv_loss = criterion_bce(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                g_adv_weight = 0.1
            else:
                g_adv_loss = torch.tensor(0.0, device=img_t.device)
                g_adv_weight = 0.0
            
            # ----- ROI-ONLY Image Loss -----
            # Loss is computed ONLY on tumor+peritumoral voxels.
            # Background voxels contribute zero gradient so the model
            # cannot ignore the tumor by predicting delta=0 everywhere.
            tumor_mask = (seg_next_gt > 0).float().unsqueeze(1)  # (B,1,D,H,W)
            dilated_mask = F.max_pool3d(tumor_mask, kernel_size=11, stride=1, padding=5)
            n_vox = dilated_mask.sum().clamp(min=1.0)

            diff_img  = img_next_pred - img_next_gt
            recon_img_mse = (dilated_mask * diff_img ** 2).sum() / n_vox
            recon_img_l1  = (dilated_mask * diff_img.abs()).sum() / n_vox

            delta_gt     = img_next_gt - img_t
            recon_delta  = (dilated_mask * (delta_pred - delta_gt).abs()).sum() / n_vox

            recon_img_loss = recon_img_mse + recon_img_l1 + recon_delta

            # ----- Seg Loss: Focal + FG-Dice -----
            recon_seg_focal = criterion_focal(seg_next_pred, seg_next_gt)
            recon_seg_dice  = criterion_dice(seg_next_pred, seg_next_gt)
            recon_seg_loss  = recon_seg_focal + recon_seg_dice
            
            g_loss = recon_img_loss + 1.0 * recon_seg_loss + g_adv_weight * g_adv_loss
            
            accelerator.backward(g_loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt_g.step()
            
            discriminator.train()
            for p in discriminator.parameters():
                p.requires_grad = True
            
            epoch_g_loss += g_loss.item()
            epoch_img_loss += recon_img_loss.item()
            epoch_seg_loss += recon_seg_loss.item()
            
            if batch_idx % 5 == 0:
                accelerator.print(
                    f"E[{epoch+1}/{epochs}] S[{batch_idx}/{len(train_loader)}] "
                    f"L_img:{recon_img_loss.item():.4f} L_seg:{recon_seg_loss.item():.4f} "
                    f"L_adv:{g_adv_loss.item():.4f}")
        
        n_steps = max(1, len(train_loader))
        
        accelerator.print(
            f"Epoch {epoch+1} | LR: {scheduler_g.get_last_lr()[0]:.6f} | "
            f"Avg G: {epoch_g_loss/n_steps:.4f} | "
            f"Avg Img: {epoch_img_loss/n_steps:.4f} | "
            f"Avg Seg: {epoch_seg_loss/n_steps:.4f}")
        
        # ==================== VALIDATION ====================
        if len(val_dataset) > 0:
            model.eval()
            val_img_loss = 0.0
            val_seg_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    img_t = batch['image_t']
                    seg_t = batch['seg_t']
                    img_next_gt = batch['image_next']
                    seg_next_gt = batch['seg_next']
                    dt = batch['dt'].unsqueeze(-1) / 365.0
                    treatment = batch['treatment']
                    
                    seg_t_onehot = F.one_hot(seg_t.long(), num_classes=4).permute(0, 4, 1, 2, 3).float()
                    model_input_t = torch.cat([img_t, seg_t_onehot], dim=1)
                    
                    delta_pred, seg_pred, _, _ = model(model_input_t, dt, treatment)
                    img_pred = img_t + delta_pred
                    
                    tumor_mask_v = (seg_next_gt > 0).float().unsqueeze(1)
                    dilated_mask_v = F.max_pool3d(tumor_mask_v, kernel_size=11, stride=1, padding=5)
                    n_vox_v = dilated_mask_v.sum().clamp(min=1.0)
                    roi_mse = ((dilated_mask_v * (img_pred - img_next_gt) ** 2).sum() / n_vox_v).item()
                    
                    full_mse = F.mse_loss(img_pred, img_next_gt).item()
                    v_seg = criterion_focal(seg_pred, seg_next_gt) + criterion_dice(seg_pred, seg_next_gt)
                    val_img_loss += roi_mse
                    val_seg_loss += v_seg.item()
                    
            n_val_steps = max(1, len(val_loader))
            accelerator.print(
                f"  Val | ROI MSE: {val_img_loss/n_val_steps:.4f} | "
                f"Seg(Focal+Dice): {val_seg_loss/n_val_steps:.4f}")
        # Step schedulers AFTER all optimizer.step() calls in this epoch
        scheduler_g.step()
        scheduler_d.step()
        
        # ==================== CHECKPOINT ====================
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and (epoch + 1) % 10 == 0:
            ckpt_dir = "checkpoints"
            os.makedirs(ckpt_dir, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_disc = accelerator.unwrap_model(discriminator)
            
            # Save model weights (for evaluate.py)
            torch.save(unwrapped_model.state_dict(),
                       os.path.join(ckpt_dir, f"timebrainsgen_v15_epoch_{epoch+1}.pth"))
            
            # Save full training state (for resume)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_g_state_dict': opt_g.state_dict(),
                'optimizer_d_state_dict': opt_d.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'scheduler_d_state_dict': scheduler_d.state_dict(),
            }, os.path.join(ckpt_dir, f"timebrainsgen_v15_resume_epoch_{epoch+1}.pth"))
            
            accelerator.print(f"Checkpoint saved at epoch {epoch+1}.")

if __name__ == "__main__":
    main()
