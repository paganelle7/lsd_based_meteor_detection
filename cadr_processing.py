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
    
def LULU(data, size = 2):
    m = torch.nn.ZeroPad2d(size-1)
    u_data = m(data).unfold(0, size*2-1, 1 ).unfold(1, size*2-1, 1)
    L = torch.concat((u_data[:,:,:size,:size].min(2)[0].min(2)[0].unsqueeze(0), u_data[:,:,:size,size-1:].min(2)[0].min(2)[0].unsqueeze(0),u_data[:,:,size-1:,:size].min(2)[0].min(2)[0].unsqueeze(0),u_data[:,:,size-1:,size-1:].min(2)[0].min(2)[0].unsqueeze(0))).max(0)[0]
    u_data = m(L).clone().unfold(0, size*2-1, 1).unfold(1, size*2-1, 1)
    U = torch.concat((u_data[:,:,:size,:size].max(2)[0].max(2)[0].unsqueeze(0), u_data[:,:,:size,size-1:].max(2)[0].max(2)[0].unsqueeze(0),u_data[:,:,size-1:,:size].max(2)[0].max(2)[0].unsqueeze(0),u_data[:,:,size-1:,size-1:].max(2)[0].max(2)[0].unsqueeze(0))).min(0)[0]
    return U

def median_torch(data, size=7):
    m = torch.nn.ZeroPad2d(int((size-1)/2))
    
    u_data = m(data).unfold(0, size, 1).unfold(1, size, 1)
    u_data = u_data.reshape((u_data.shape[0], u_data.shape[1], size**2))
    # print(u_data.shape)
    return u_data.sort(-1)[0][:,:,int((size**2-1)/2)]

def denoising_torch(data, LULU_size, meadian_filter_size):
    return median_torch(LULU(data, size=LULU_size), size=meadian_filter_size)

def epiphania(data, size=3):
    m = torch.nn.ZeroPad2d(int((size-1)/2))
    step = size*2 - 3 
    u_data = m(data).unfold(0, size, 1).unfold(1, size, 1)
    test = torch.cat([u_data[:,:,0], u_data[:,:,1:-1,-1],u_data[:,:,-1].flip(-1),u_data[:,:,1:-1,0].flip(-1)],dim=2) 
    tt = torch.cat([ test.roll(step,-1) + test, test.roll(step+1,-1) + test, test.roll(step+2,-1) + test ], dim=2)/2
    mt = tt.max(dim=-1)[0]
    return torch.where(data>mt, data, mt)