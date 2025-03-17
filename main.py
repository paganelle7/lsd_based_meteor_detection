import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import meteor_segment_detection_Copy1
import cadr_processing
import search_meteors_from_segments


# data1
#denoising
background_kernel = torch.ones((3,3),device=data1.device)
data1 = torch.nn.functional.max_pool2d(data1[None,None,:,:], 3, stride=1, padding=1)[0,0]
data1 = cadr_processing.median_torch(data1, size=7)
data1 = torch.nn.functional.conv2d(data1[None,None,:,:], background_kernel.unsqueeze(0).unsqueeze(0), padding='same')[0,0] 

#line segment detection
hweight_of_groops,hx_center, hy_center, x_end, y_end, hgrad_x, hgrad_y, w2,x2,y2,xe2, ye2, m2x,m2y = meteor_segment_detection_Copy1.lsd_meteor_on_image(data1, threshold=0.8,batch_size=50,weight_un=15)


#groop concatenating
minimal_groop = 50
wight_threshold = 15
len_comparing=0.4

num, la, ra  = search_meteors_from_segments.endge_comparing(hweight_of_groops,hx_center, hy_center, x_end, y_end, hgrad_x, hgrad_y, minimal_groop,wight_threshold,len_comparing)
num2, la2, ra2  = search_meteors_from_segments.endge_comparing(w2,x2,y2,xe2, ye2, m2x,m2y, minimal_groop,wight_threshold,len_comparing)

print('Meteors founded = ', num+num2)