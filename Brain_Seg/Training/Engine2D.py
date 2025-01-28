import torch
import tqdm
from monai.networks.utils import one_hot
import matplotlib.pyplot as plt
from torch.nn.functional import softmax
import seaborn as sns
from utils import evaluate, set_seed, get_metrics
from configs import config
from Dataset.Transforms2D import padder
import os
import nibabel as nib
import numpy as np

class Engine:
    def __init__(self, model, loss, optimizer, train_loader, val_loader,test_loader, scheduler, scheduler_step="epoch", device=None):
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
                images = batch["image"].to(self.device)
                labels = batch["mask"].to(self.device)
                labels=one_hot(labels,num_classes=4,dim=1).to(self.device)
                preds = self.model(images)
                loss = self.loss(preds, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                preds = preds.detach().cpu()
                labels = labels.cpu()
                evaluate(preds, labels)
                if self.scheduler and self.scheduler_step == 'batch':
                    self.scheduler.step()

            epoch_loss /= len(self.train_loader)
            Dice_score, Mean_Iou = get_metrics()
            val_loss, val_Dice_score, val_Mean_Iou = self.validate()

            print(f"\nTraining: Loss = {epoch_loss:.4f}, Dice_score = {Dice_score.numpy()}, Mean_Iou = {Mean_Iou.numpy()}, AVG_Dice = {Dice_score.mean().item()}, AVG_Iou = {Mean_Iou.mean().item()}")
            print(f"Validation: Loss = {val_loss:.4f}, Dice_score = {val_Dice_score.numpy()}, Mean_Iou = {val_Mean_Iou.numpy()}, AVG_Dice = {val_Dice_score.mean().item()}, AVG_Iou = {val_Mean_Iou.mean().item()}")
            results['train']['loss'].append(epoch_loss)
            results['train']['Dice_score'].append(Dice_score)
            results['train']['Mean_Iou'].append(Mean_Iou)
            results['val']['loss'].append(val_loss)
            results['val']['Dice_score'].append(val_Dice_score)
            results['val']['Mean_Iou'].append(val_Mean_Iou)

            if self.scheduler and self.scheduler_step == 'epoch':
                self.scheduler.step()

            if val_Dice_score.mean() > best_Dice_score:
                best_Dice_score = val_Dice_score.mean()
                results["best_epoch"] = epoch + 1
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_Dice_score': best_Dice_score,
                    'results': results
                    }, f"best_checkpoint_basic_2D_{epoch+1}.pth")
            print(f"==============================================================")
        return results

    @torch.no_grad
    def validate(self, data_loader=None):
      data_loader = data_loader or self.val_loader
      self.model.eval()
      total_loss = 0
      total_dices = []
      total_ious = []
      slices_per_volume = 128 -2

      accumulated_images = []
      accumulated_labels = []

      for batch in tqdm.tqdm(data_loader):
          images = batch["image"].to(self.device)
          labels = batch["mask"].to(self.device)
          labels = one_hot(labels, num_classes=4).to(self.device)

          preds = self.model(images)
          loss = self.loss(preds, labels)
          total_loss += loss.item()
  
          accumulated_images.append(preds.detach().cpu())
          accumulated_labels.append(labels.cpu())

          if len(accumulated_images) >= slices_per_volume:

              volume_preds = torch.stack(accumulated_images[:slices_per_volume], dim=3)
              volume_labels = torch.stack(accumulated_labels[:slices_per_volume], dim=3)
              
              volume_preds = padder(volume_preds[0]).unsqueeze(0)
              volume_labels = padder(volume_labels[0]).unsqueeze(0)

              accumulated_images = accumulated_images[slices_per_volume:]
              accumulated_labels = accumulated_labels[slices_per_volume:]

              evaluate(volume_preds, volume_labels)
              dice, iou = get_metrics()
              total_dices.append(dice)
              total_ious.append(iou)

      avg_loss = total_loss / len(data_loader)
      avg_dice = torch.stack(total_dices).mean(dim=0)
      avg_iou = torch.stack(total_ious).mean(dim=0)

      return avg_loss, avg_dice, avg_iou

    @torch.no_grad
    def test_and_save_predictions(self, data_loader, output_dir):
        self.model.eval()
        slices_per_volume = 128-2  
        accumulated_images = []
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm.tqdm(data_loader)):
                images = batch["image"].to(self.device)
                affine=batch["affine"]
                preds = self.model(images)
                accumulated_images.append(preds.detach().cpu())
                if len(accumulated_images) >= slices_per_volume:
                    volume_preds = torch.stack(accumulated_images[:slices_per_volume], dim=3)                    
                    volume_preds = padder(volume_preds[0]).unsqueeze(0)
                    volume_preds=torch.argmax(volume_preds,dim=1).squeeze().squeeze()
                    accumulated_images = accumulated_images[slices_per_volume:]
                    output_path = os.path.join(output_dir, f"prediction_{batch_idx}.nii.gz")
                    nifti_image = nib.Nifti1Image(volume_preds.astype("int16"), affine=affine[0])
                    nib.save(nifti_image, output_path)
                    print(f"Saved prediction volume to {output_path}")
            if accumulated_images:
                print("Warning: Remaining slices were not saved as a complete volume.")

