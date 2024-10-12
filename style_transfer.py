
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


def load_image(img_path, max_size=500, shape=None):
    ''' Load in and transform an image, making sure the image is <= 400 pixels in the x-y dims.'''
    
    image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape
        
    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    
    return image



def im_convert(tensor):
    ''' helper function for un-normalizing an image and converting it from a Tensor image to a NumPy image for display '''
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def imshow(img):              
    ''' Pour afficher une image '''
    plt.figure(1)
    plt.imshow(img)
    plt.show()






################################################### VGG FEATURES #####################################################
def get_features(image, model, layers=None):
    ''' 
        Run an image forward through a model and get the features for a set of layers
    '''
    
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv0',
                  '1': 'conv1',
                  '2': 'conv2',
                  '3': 'conv3',
                  '4': 'conv4',
                  '5': 'conv5', 
                  '10': 'conv10', 
                  '19': 'conv19',   ## content representation
                }

    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x            
    return features



def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    # get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    
    # reshape so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    
    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())
    
    return gram




if __name__ == '__main__':

    current_time = datetime.now().strftime("%d_%H_%M")
    writer = SummaryWriter(f'runs/style_transfer_{current_time}')

    ##########################" VGG "#########################################################""
    # get the "features" portion of VGG19 (we will not need the "classifier" portion)
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

    # freeze all VGG parameters since we're only optimizing the target image
    for param in vgg.parameters():
        param.requires_grad_(False)


    # move the model to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    vgg.to(device)

    features = list(vgg)[:23]
    for i,layer in enumerate(features):
        print(i,"   ",layer)


    ########################## DISPLAY IMAGE#########################################################""
    content = load_image('data/i.jpg').to(device)
    style = load_image('data/s.jpg', shape=content.shape[-2:]).to(device)

    # imshow(im_convert(content))
    # imshow(im_convert(style))

    # _, d, h, w = content.size()
    # print("content size=",d,h,w)
    # _, d, h, w = style.size()
    # print("style size=",d,h,w)


    # get content and style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    print(type(content_features))
    for key, value in content_features.items():
        # print("key=",key)
        # print("value shape=", type(value))
        vnp = value.to("cpu").clone().detach().numpy()
        # print("value shape=", vnp.shape)



    target = content.clone().requires_grad_(True).to(device)



    show_every = 100
    save_every = 1000
    optimizer = optim.Adam([target], lr=0.003)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}           # dict with each gram matrix for each feature name
    style_layers = {  'conv0', 'conv5', 'conv10', }

    for i in range(20001):
        content_features = get_features(target, vgg)
        style_features = get_features(target, vgg)
        target_features = get_features(target, vgg)
    
        content_loss = torch.mean((target_features['conv19'] - content_features['conv19'])**2)
        
        style_loss = 0
        target_grams = {layer: gram_matrix(target_features[layer]) for layer in target_features}           # dict with each gram matrix for each feature name
        #for key, value in style_features.items():
        for key in style_layers:
            #d, hw = style_grams[key].shape
            style_loss  +=  torch.mean((style_grams[key] - target_grams[key])**2)  #/ (d * hw)
        style_loss /= len(style_layers)

        
        total_loss =  0.5*content_loss + 0.5*style_loss
    
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        
        if  i % show_every == 0:
            print('Total loss: ', i, total_loss.item())
            writer.add_scalar('Loss/total', total_loss.item(), i)
            if i % save_every == 0:
                plt.imsave("data/output"+str(i)+".png", im_convert(target))
                writer.add_image('Generated Image', torch.tensor(im_convert(target)), global_step=i, dataformats='HWC')
                print("save %d" % i)
        
    writer.close()