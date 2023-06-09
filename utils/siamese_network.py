import os
import torch
import torch.nn as nn
from tqdm import tqdm 
from utils.loss_metrics import RunningMetric
from torchvision.models import vgg16, VGG16_Weights, resnet50, ResNet50_Weights


#create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity detection.
    
    Args:
        nn.Module: Pytorch neural network module.
    """

    def __init__(self, model_name='vgg16', device='cpu'):
        """
        init function for the Siamese Network.
        Args:
            model_name (str): Name of the model to use default: "vgg16", option = "resnet50"
        Returns:
            Class object.
        """
        #super(self,SiameseNetwork).__init__() # inherit from nn.Module class
        super().__init__() # inherit from nn.Module class
        self.device = device
        
        if model_name == 'vgg16':      
            weights = VGG16_Weights.DEFAULT
            self.model = vgg16(weights=weights)
            self.model.classifier[6] = nn.Linear(in_features = self.model.classifier[6].in_features, out_features=4)
        elif model_name == 'resnet50':
            weights = ResNet50_Weights.DEFAULT
            self.model = resnet50(weights=weights)
            self.model.fc = nn.Linear(in_features = self.model.fc.in_features, out_features=4)
        
        self.counter    = []
        self.acc_train  = []
        self.acc_val    = []
        self.loss_train = []
        self.loss_val   = []
        
    def forward_once(self, x):
        """
        This function passes the input through the network once.
        It's output is used to determine the similiarity
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, channels, height, width)
        
        Returns:
            output (torch.Tensor): Output tensor with shape (batch_size, embedding_size)
        
        """
        output =self.model(x)
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
    
    def train_eval(self,model,optimizer,criterion,dataloaders,num_epochs = 10):
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
            for phase in ['train','val']:
                if phase == 'train':
                    model.train()
                    datos=train_dataloader
                else: 
                    model.eval()
                    datos = val_dataloader
                
                running_loss = RunningMetric()  #perdida tasa de error de la error
                running_acc = RunningMetric()   #precision
                              
                label_batches = torch.ones(64).to(self.device)
                
                # Number of batches in one epoch (train or val)
                total_batches = len(datos)
                                
                for l, (img0, img1, label) in tqdm(enumerate(datos,0), total=total_batches, desc=f'Epoch {epoch + 1}/{num_epochs} - {phase}'):
                    
                    img0, img1, label = img0.to(self.device),img1.to(self.device), label.to(self.device)
                    
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                            
                            # Forward pass
                            output1, output2 = model(img0, img1)
                            
                            #preds_cnn1-2 are the predicted classes for the two images
                            _,preds_cnn1 = torch.max(output1,1)
                            _,preds_cnn2 = torch.max(output2,1)
                            
                            label_p = torch.squeeze(label)
                            preds_siamese=torch.ones(64).to(self.device)
       
                            # Compare the two outputs and determine if they are images of the same class
                            for idx, val in enumerate(preds_cnn1):
                                label_batches[idx]=label_p[idx]
                                if preds_cnn2[idx]==preds_cnn1[idx]:
                                    preds_siamese[idx]=0
                                if preds_cnn2[idx]!=preds_cnn1[idx]:
                                    preds_siamese[idx]=1

                            # Calculate the loss
                            loss_contrastive = criterion(output1, output2, label)
                            
                            if phase =='train':
                                # Calculate the backpropagation
                                loss_contrastive.backward()
                                optimizer.step()
                        
                    batch_size=img0.size()[0]                 
                    running_loss.update(loss_contrastive.item()*batch_size,batch_size)
                    running_acc.update(torch.sum(preds_siamese==label_batches).float(),batch_size)
                        
                    if (l+1) / total_batches == 1:
                        if(phase=='train'):
                            train_acc = running_acc().cpu().numpy()
                            train_loss = running_loss()
                            self.counter.append(epoch)
                            self.acc_train.append(train_acc)
                            self.loss_train.append(train_loss)

                        elif(phase=='val'):
                            val_acc = running_acc().cpu().numpy()
                            val_loss = running_loss()
                            self.acc_val.append(val_acc)
                            self.loss_val.append(val_loss)                                       
            # Print metrics and progress bar
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Training acc: {train_acc:.3f} - Training loss: {train_loss:.3f} - Validation acc: {val_acc:.3f} - Validation loss: {val_loss:.3f}")
        
        metrics = {
            'counter': self.counter, 
            'acc_train': self.acc_train,
            'loss_train': self.loss_train,
            'acc_val': self.acc_val,
            'loss_val': self.loss_val}
        
        return model, metrics