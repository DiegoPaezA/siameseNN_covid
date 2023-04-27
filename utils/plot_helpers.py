import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import torch



def visualize_model(model, dataloaders, device, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(preds[j]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# Creating some helper functions
def imshow(img, text=None):
    
    npimg = img.cpu().numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show();    

def show_plot(iteration,loss, name:str, figsize=(5,5)):
    plt.figure(figsize=figsize)
    plt.plot(iteration,loss)
    plt.title(f"{name} at epoch {len(iteration)}")
    plt.show();
    
def plot_training_curves(history:dict):
    """
    Plot training curves for accuracy and loss metrics.
    Args: 
        history (dict): Dictionary with the training and validation metrics.
    return:
        None


    """
    # Plot training & validation accuracy values

    loss = history['loss_train']
    val_loss = history['loss_val']
    acc = history['acc_train']
    val_acc = history['acc_val']
    
    plt.subplot(2, 1, 1)
    plt.plot(acc)
    plt.plot(val_acc)
    plt.title('Model Training Curves')
    plt.ylabel('Accuracy')
    #plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    plt.grid(True)
    # Plot training & validation loss values
    plt.subplot(2, 1, 2)
    plt.plot(loss)
    plt.plot(val_loss)
    #plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.grid(True)
    plt.show()

def compare_images(image_1:torch.tensor, image_2:torch.tensor, class_name:list, text:str=None):
    """
    Compare two images and print the class of each image.
    Args:
        image_1 (torch.tensor): Pytorch tensor with the first image.
        image_2 (torch.tensor): Pytorch tensor with the second image.
        class_name (list): List with the class names.
    Returns:
        None
    """
    image_1 = image_1.cpu().numpy()
    image_2 = image_2.cpu().numpy()
    
    fig, ax = plt.subplots(1, 2, figsize=(8, 8))
    ax[0].imshow(np.transpose(image_1,(1, 2, 0)))
    ax[0].set_title(f"Class: {class_name[0]}")
    ax[0].axis('off')
    ax[1].imshow(np.transpose(image_2,(1, 2, 0)))
    ax[1].set_title(f"Class: {class_name[1]}")
    ax[1].axis('off')
    
    plt.tight_layout()
    if text:
        plt.text(-220, 18, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})