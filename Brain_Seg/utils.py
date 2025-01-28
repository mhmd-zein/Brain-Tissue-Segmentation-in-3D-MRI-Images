from monai.metrics import DiceMetric,MeanIoU
from monai.networks.utils import one_hot
import torch, random, numpy as np, monai
import torch


diceclass = DiceMetric(include_background=False, ignore_empty=True, reduction='mean_batch')
iouclass = MeanIoU(include_background=False, ignore_empty=True, reduction='mean_batch')

def evaluate(preds, labels): 
    
    print(torch.argmax(preds,1,keepdim=True).shape, torch.argmax(labels,1,keepdim=True).shape)
    print(torch.argmax(preds,1,keepdim=True).unique(), torch.argmax(labels,1,keepdim=True).unique())
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title("Pred")
    plt.imshow(torch.argmax(preds,1,keepdim=True)[0,0,:,:,torch.argmax(preds,1,keepdim=True).shape[-1]//2])
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('GT')
    plt.imshow(torch.argmax(labels,1,keepdim=True)[0,0,:,:,torch.argmax(labels,1,keepdim=True).shape[-1]//2])
    plt.tight_layout()
    plt.show()

    preds = one_hot(torch.argmax(preds,1,keepdim=True),num_classes=4)
    diceclass(preds, labels)
    iouclass(preds, labels)


def get_metrics():
  iou = iouclass.aggregate()
  iouclass.reset()
  dice = diceclass.aggregate()
  diceclass.reset()
  return dice, iou

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.set_determinism(seed=seed)