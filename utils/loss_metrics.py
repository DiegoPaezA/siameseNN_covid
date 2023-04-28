"""
This file contains the loss and metrics functions used in the training and validation of the model.
"""
import torch
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    """
    This function implements the contrastive loss function.
    Args:
        margin (float): Margin to be used for the loss calculation.
    Returns:
        torch.Tensor: Contrastive loss value.
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
  
class RunningMetric():
    """
    This functions are used to calculate the running average of the loss and accuracy.
    """
    def __init__(self):
        self.S = 0
        self.N = 0
    
    def update(self, val_, size):
        self.S += val_
        self.N += size
    
    def __call__(self):
        return self.S/float(self.N)