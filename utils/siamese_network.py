import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import torch.nn as nn


#create the Siamese Neural Network
class SiameseNetwork(nn.Module):
    """
    Siamese Network for image similarity detection.
    
    Args:
        nn.Module: Pytorch neural network module.
    Returns:
        None
        
    
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.Vgg11=models.vgg11(pretrained=True)
        num_ftrs = self.Vgg11.classifier[6].in_features
        self.Vgg11.classifier[6] = nn.Linear(num_ftrs, 4)
        
        self.counter = []
        self.loss_history = []
        self.acc =  []
        self.loss = []
        
    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        #output = self.cnn1(x)
        #output = output.view(output.size()[0], -1)
        #output = self.fc1(output)
        output =self.Vgg11(x)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    
    def train_eval(self,model,optimizer,criterion,dataloaders,num_epochs = 70):
        # Iterate throught the epochs
        train_dataloader=dataloaders['train']
        val_dataloader=dataloaders['val']
        iteration_number=0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-'*10)
            y=0
            for phase in ['train','val']:
                if phase == 'train':
                    model.train()
                    datos=train_dataloader
                else: 
                    model.eval()
                    datos=val_dataloader
                # Iterate over batches
                running_loss=RunningMetric()  #perdida tasa de error de la error
                running_acc=RunningMetric()   #precision
                
                #labelPT=np.ones(shape=(1,64))
                labelPT=torch.ones(64) # 64 is the batch size
                    #for i, (img0, img1, label) in enumerate(train_dataloader,0):
                for l, (img0, img1, label) in enumerate(datos,0):
                    img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()
                    optimizer.zero_grad()     #llevar a cero..reiniciar
                    with torch.set_grad_enabled(phase=='train'):
                            # Pass in the two images into the network and obtain two outputs
                            print(f"img0 {img0.shape}")
                            output1, output2 = model(img0, img1)
                            
                            #preds1-2 is the index of the max value of output1-2 for each image.
                            _,preds1=torch.max(output1,1) 
                            _,preds2=torch.max(output2,1) 
                            
                            labelP = torch.squeeze(label)

                            preds3T=torch.ones(64) # 64 is the batch size
       
                            # Compare the two outputs and determine if they are similar or not
                            for idx, val in enumerate(preds1):
                                labelPT[idx]=labelP[idx]
                                if preds2[idx]==preds1[idx]:
                                    preds3T[idx]=0
                                if preds2[idx]!=preds1[idx]:
                                    preds3T[idx]=1
                            
                            print('preds3T: ',preds3T)

                            # Pass the outputs of the networks and label into the loss function
                            loss_contrastive = criterion(output1, output2, label)
                            if phase =='train':
                                # Calculate the backpropagation
                                loss_contrastive.backward()
                                # Optimize
                                optimizer.step()
                        
                    batch_size=img0.size()[0]
                    running_loss.update(loss_contrastive.item()*batch_size,batch_size)
                                              

                    running_acc.update(torch.sum(preds3T==labelPT).float(),batch_size)
                        
                    if l % 30 == 0  :
                        # print('preds3TT',preds3TT)
                        # print('labelPT_tensor',labelPT_tensor)
                        print("Loss: {:.4f} Acc: {:.4f} ".format(running_loss(),running_acc()))
                        print(f"Epoch number {epoch}/{num_epochs} Current loss {loss_contrastive.item()}")
                        iteration_number += 10
                        self.counter.append(iteration_number)
                        self.loss_history.append(loss_contrastive.item())
                        self.acc.append(running_acc())
                        self.loss.append(running_loss())
                        # print('counter =',self.counter)
                        # print('loss_history =',self.loss_history)
                        # print('loss =',self.loss)
                        # print('Acc =',self.acc)
        metrics = {
            'counter': self.counter, 
            'loss_history': self.loss_history, 
            'acc': self.acc, 
            'loss': self.loss}
        return model, metrics

# Define the Contrastive Loss Function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
      # Calculate the euclidian distance and calculate the contrastive loss
      euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)

      loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                    (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


      return loss_contrastive
  
class RunningMetric():      #CALCULAS PROMEDIOS EN EL TIEMPO
  def __init__(self):
    self.S = 0
    self.N = 0
    
  def update(self, val_, size):
    self.S += val_
    self.N += size
    
  def __call__(self):
    return self.S/float(self.N)