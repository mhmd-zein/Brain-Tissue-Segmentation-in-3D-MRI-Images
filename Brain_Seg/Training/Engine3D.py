import torch
from torch._prims_common import Dim
from torch.functional import _return_counts
import tqdm
from monai.networks.utils import one_hot
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import seaborn as sns
from utils import evaluate, set_seed, get_metrics
from configs import config
from Dataset.Transforms2D import padder
from monai.inferers import sliding_window_inference
from monai.data import GridPatchDataset, DataLoader, PatchIterd
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from torch.utils.data import RandomSampler
from monai.transforms import AsDiscrete
import os
import torch
import tqdm
import nibabel as nib
import numpy as np

class Engine:
    def __init__(self, model, loss, optimizer, train_loader, val_loader, test_loader,scheduler, scheduler_step="epoch", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss = loss.to(self.device)
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.scheduler = scheduler
        self.scheduler_step = scheduler_step
        self.test_loader=test_loader

    def train(self,starting_epoch=0, epochs=1, results=None):
        results = results or {
            "best_epoch": -1,
            "train": {"loss": [], "Dice_score": [], "Mean_Iou": []},
            "val": {"loss": [], "Dice_score": [], "Mean_Iou": []}
        }
        best_Dice_score = 0
        i=0
        for epoch in range(starting_epoch,epochs):
            print(f"================== Epoch {epoch + 1} / {epochs} ==================")
            epoch_loss = 0
            all_labels = []
            all_preds = []
            Dice_scores=[]
            mean_ious=[]
            self.model.train()
            for i, batch in enumerate(tqdm.tqdm(self.train_loader)):
                image = batch['image'].to(self.device)
                mask = batch['mask']
                mask = one_hot(mask, num_classes=4, dim=1).to(self.device)              
                preds = self.model(image)
                loss = self.loss(preds, mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()           
                preds = preds.detach().cpu()
                mask = mask.cpu()
                evaluate(preds, mask)                
                if self.scheduler and self.scheduler_step == 'batch':
                    self.scheduler.step()

            epoch_loss /= len(self.train_loader)
            Dice_score, Mean_Iou = get_metrics()
            if (epoch + 1) % 5 == 0:
                val_loss, val_Dice_score, val_Mean_Iou = self.validate()
                print(f"Validation: Loss = {val_loss:.4f}, Dice_score = {val_Dice_score.numpy()}, Mean_Iou = {val_Mean_Iou.numpy()}, AVG_Dice = {val_Dice_score.mean().item()}, AVG_Iou = {val_Mean_Iou.mean().item()}")
                results['val']['loss'].append(val_loss)
                results['val']['Dice_score'].append(val_Dice_score)
                results['val']['Mean_Iou'].append(val_Mean_Iou)

                if val_Dice_score.mean() > best_Dice_score:
                    best_Dice_score = val_Dice_score.mean()
                    results["best_epoch"] = epoch + 1
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_Dice_score': best_Dice_score,
                        'results': results
                    }, f"best_checkpoint_patchbased_2_{epoch+1}.pth")
            print(f"Training: Loss = {epoch_loss:.4f}, Dice_score = {Dice_score.numpy()}, Mean_Iou = {Mean_Iou.numpy()}, AVG_Dice = {Dice_score.mean().item()}, AVG_Iou = {Mean_Iou.mean().item()}")
            results['train']['loss'].append(epoch_loss)
            results['train']['Dice_score'].append(Dice_score)
            results['train']['Mean_Iou'].append(Mean_Iou)

            if self.scheduler and self.scheduler_step == 'epoch':
                self.scheduler.step()

            print(f"==============================================================")
        return results
        
    @torch.no_grad()
    def validate(self, data_loader=None):
        data_loader = data_loader or self.val_loader
        self.model.eval()
        total_loss = 0
        all_dices = []
        all_ious = []
        with torch.no_grad():
          for batch in tqdm.tqdm(data_loader):
              image = batch["image"].to(self.device)
              mask = batch["mask"]
              labels = one_hot(mask, num_classes=4).to(self.device)
              preds = sliding_window_inference(image, (64, 64, 64), 1, self.model)
              loss = self.loss(preds, labels)
              total_loss += loss.item()
              preds = preds.detach().cpu()
              labels = labels.cpu()
              evaluate(preds, labels)
              dice_score, iou_score = get_metrics()
              all_dices.append(dice_score)
              all_ious.append(iou_score)
          total_loss /= len(data_loader)
          avg_dice = torch.stack(all_dices).mean(dim=0)
          avg_iou = torch.stack(all_ious).mean(dim=0)

        return total_loss, avg_dice, avg_iou

    @torch.no_grad()
    def test_and_save_predictions(self, data_loader=None, save_dir="predicted_volumes"):
        data_loader = data_loader or self.test_loader
        self.model.eval()
        
        os.makedirs(save_dir, exist_ok=True)  
        
        with torch.no_grad():
            for idx, batch in enumerate(tqdm.tqdm(data_loader)):
                image = batch["image"].to(self.device)
                prefix=batch["prefix"]
                preds = sliding_window_inference(image, (64, 64, 64), 1, self.model)
                preds = preds.argmax(dim=1).detach().cpu().squeeze(0).numpy() 
                preds = preds.astype(np.int32)
                affine = batch.get("affine", torch.eye(4)).numpy() 
                nifti_img = nib.Nifti1Image(preds, affine)
                file_name = f"{prefix}.nii.gz"
                nib.save(nifti_img, os.path.join(save_dir, file_name))     
        print(f"All predictions saved in {save_dir}")



