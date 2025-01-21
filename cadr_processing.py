import torch
from numpy import exp, pi

def gauss_kernel(size, sigma, device):
    # pi = 3.14159265358979323846
    data = torch.zeros((size,size),device=device)
    center = size/2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] += 1 / 2 / pi / (sigma ** 2) * exp(-1 / 2 * ((i - center +0.5) ** 2 + (j - center +0.5) ** 2) / (sigma ** 2))
    return data

def get_slice(image_sequence, background_kernel, image_kernel, cadr_in_work=10):
    seq_shape = image_sequence.shape
    shape = (seq_shape[2], seq_shape[3])
    length = shape[0]
    background = image_sequence[:-cadr_in_work].mean(0)
    
    background_0  = torch.nn.functional.conv2d(background[None,0,:,:], background_kernel.unsqueeze(0).unsqueeze(0), padding='same')[0]
    background_1  = torch.nn.functional.conv2d(background[None,1,:,:], background_kernel.unsqueeze(0).unsqueeze(0), padding='same')[0]
    background_2  = torch.nn.functional.conv2d(background[None,2,:,:], background_kernel.unsqueeze(0).unsqueeze(0), padding='same')[0]
    background = torch.stack((background_0,background_1,background_2))
    
    current_processing = image_sequence[-cadr_in_work:] / (background+1)
    
    ready_for_grad_image = torch.max(current_processing, dim=0)[0]
    ready_for_grad_image -= torch.min(ready_for_grad_image)
    ready_for_grad_image = torch.pow(ready_for_grad_image[0]*ready_for_grad_image[1]*ready_for_grad_image[2],1/3)#ready_for_grad_image.mean(0)
    ready_for_grad_image = torch.nn.functional.conv2d(ready_for_grad_image[None,None,:,:], image_kernel.unsqueeze(0).unsqueeze(0), padding='same')[0,0]

    return ready_for_grad_image
    