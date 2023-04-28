import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

import torch.nn as nn


#create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity detection.
    
    Args:
        nn.Module: Pytorch neural network module.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Setting up the Sequential of CNN Layers
        weights = VGG16_Weights.DEFAULT
        self.Vgg16 = vgg16(weights=weights)
        #num_ftrs = self.Vgg16.classifier[6].in_features
        #self.Vgg16.classifier[6] = nn.Linear(num_ftrs, 4)
        self.Vgg16.classifier[6] = nn.Linear(in_features = self.Vgg16.classifier[6].in_features, out_features=4)
        
        self.counter = []
        self.acc_train =  []
        self.acc_val =  []
        self.loss_train = []
        self.loss_val = []
        
    def forward_once(self, x):
        """
        This function passes the input through the network once.
        It's output is used to determine the similiarity
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
        
        Returns:
            output (torch.Tensor): Output tensor with shape (batch_size, embedding_size)
        
        """
        output =self.Vgg16(x)
        return output

    def forward(self, input1, input2):
        """
        In this function we pass in both images and obtain both vectors
        which are returned
        
        Args:
            input1 (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
            input2 (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
        Returns:
            output1 (torch.Tensor): Output tensor with shape (batch_size, embedding_size)
            output2 (torch.Tensor): Output tensor with shape (batch_size, embedding_size)
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    
    def train_eval(self,model,optimizer,criterion,dataloaders,num_epochs = 70):
        """
        This function is used to train and evaluate the Siamese Network.
        
        Args:
            model (torch.nn.Module): Pytorch neural network module.
            optimizer (torch.optim): Pytorch optimizer.
            criterion (torch.nn): Pytorch loss function.
            dataloaders (dict): Pytorch dataloaders.
            num_epochs (int): Number of epochs to train the model.
        Returns:
            Model with the best weights.
            metrics (dict): Dictionary with the training and validation metrics.
        """
        
        # Iterate throught the epochs
        train_dataloader = dataloaders['train']
        val_dataloader = dataloaders['val']
                
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-'*30)
            
            for phase in ['train','val']:
                if phase == 'train':
                    model.train()
                    datos=train_dataloader
                else: 
                    model.eval()
                    datos = val_dataloader
                # Iterate over batches
                running_loss = RunningMetric()  #perdida tasa de error de la error
                running_acc = RunningMetric()   #precision
                              
                label_batches = torch.ones(64).to(self.device)
                               
                for l, (img0, img1, label) in enumerate(datos,0):
                    # Take a batch of images and labels and send them to the device
                    img0, img1, label = img0.to(self.device),img1.to(self.device), label.to(self.device)
                    #print(f"iteracion {l+1} de {len(datos)}")
                    optimizer.zero_grad()     #llevar a cero..reiniciar
                    with torch.set_grad_enabled(phase=='train'):
                            
                            # Forward pass
                            # Pass in the two images into the network and obtain two outputs
                            output1, output2 = model(img0, img1)
                            
                            #preds_cnn1-2 is the index of the max value of output1-2 for each image.
                            _,preds_cnn1 = torch.max(output1,1)
                            _,preds_cnn2 = torch.max(output2,1)
                            
                            label_p = torch.squeeze(label)
                            preds_siamese=torch.ones(64).to(self.device) # 64 is the batch size
       
                            # Compare the two outputs and determine if they are images of the same class
                            for idx, val in enumerate(preds_cnn1):
                                label_batches[idx]=label_p[idx]
                                if preds_cnn2[idx]==preds_cnn1[idx]:
                                    preds_siamese[idx]=0
                                if preds_cnn2[idx]!=preds_cnn1[idx]:
                                    preds_siamese[idx]=1

                            # Pass the outputs of the networks and label into the loss function
                            loss_contrastive = criterion(output1, output2, label)
                            
                            if phase =='train':
                                # Calculate the backpropagation
                                loss_contrastive.backward()
                                # Optimize
                                optimizer.step()
                        
                    batch_size=img0.size()[0]                 
                    running_loss.update(loss_contrastive.item()*batch_size,batch_size)
                    running_acc.update(torch.sum(preds_siamese==label_batches).float(),batch_size)
                        
                    if (l+1) / len(datos) == 1:
                        if(phase=='train'):
                            print(f"Epoch: {epoch+1}/{num_epochs}, updated train metrics, acc: {running_acc()}, loss: {round(running_loss(),3)}")
                            self.counter.append(epoch)
                            self.acc_train.append(running_acc().cpu().numpy())
                            self.loss_train.append(running_loss())

                        elif(phase=='val'):
                            print(f"Epoch: {epoch+1}/{num_epochs}, updated val metrics, acc: {running_acc()}, loss: {round(running_loss(),3)}")
                            self.acc_val.append(running_acc().cpu().numpy())
                            self.loss_val.append(running_loss())
                            print('-'*20)
                    #print(f"Batch number {l}/{len(datos)}, len of img0 {len(img0)}, len of label {len(label)}")
        
        metrics = {
            'counter': self.counter, 
            'acc_train': self.acc_train,
            'loss_train': self.loss_train,
            'acc_val': self.acc_val,
            'loss_val': self.loss_val}
        
        return model, metrics

# Define the Contrastive Loss Function
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