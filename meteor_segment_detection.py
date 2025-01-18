import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def grad_ima(image):
    c =image[:-1, 1:] - image[:-1,:-1]
    b = image[1:, :-1]-image[:-1,:-1]
    d= image[1:,1:] - image[:-1, 1:] - image[1:, :-1] + image[:-1,:-1]
    x = b + d/2
    y = c + d/2
    r = torch.pow(torch.sqrt(x**2+y**2),1)
    
    return torch.nan_to_num(x/r),torch.nan_to_num(y/r)


def unique_neighbors(grad_x,n=8):
    device = grad_x.device
    shape = grad_x.shape[0]
    ind_arr =torch.arange(shape, device=device).int()
    ind0_arr =torch.arange(shape, device=device).int()
    weights = torch.ones(shape).int()
    for i in range(1,n):
        condition = (grad_x[i:]==grad_x[:-i])
        ind_arr[i:] = torch.minimum(ind0_arr[i:] - i*condition, ind_arr[i:])
        weights[i:] *= 1- 1*condition
        
        weights[:-i] += 1*condition*torch.sign(weights[:-i])
        
    return torch.where(weights>0, weights,0), ind_arr

def one_grad_groop(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center, threshold=0.9):
    shape = main_grad_x.shape;    h,w = main_grad_x.shape;    device = main_grad_x.device
    last_grad_x, last_grad_y = main_grad_x[0], main_grad_y[0]
    last_weight = weight_of_groops[0]#torch.ones((w)); 
    weight_of_groops[0]=0
    last_center_x, last_center_y = x_center[0], y_center[0]
    with torch.no_grad():
        for i in range(h-1):
            t0 = time()
            #Загрузка
            grad_x, grad_y = main_grad_x[i+1], main_grad_y[i+1]
            x_read = x_center[i+1]
            y_read = y_center[i+1]
            weight_of_groops[i] = 0
            #Здесь нужно определить массив для выбора лидера
            a,b,c = last_grad_x[:-1]*grad_x[1:]+ last_grad_y[:-1]*grad_y[1:], last_grad_x*grad_x+last_grad_y*grad_y, last_grad_x[1:]*grad_x[0:-1]+last_grad_y[1:]*grad_y[0:-1]
            a,c = torch.cat((torch.tensor([-1000]).to(device),a), dim=0), torch.concat((c,torch.tensor([-1000]).to(device)), dim=0)
            dot = torch.concat((a.unsqueeze(dim=1),b.unsqueeze(dim=1),c.unsqueeze(dim=1)), axis=1)
            dot, ind = torch.max(dot, axis=1)
            
            condition = dot>threshold
            index_new_connection = torch.arange(w, device=device).int() + ind-1
            
            # #Назначение лидеров групп
            leader_x = last_center_x[index_new_connection]
            last_center_x = torch.where(condition, leader_x,x_read)# torch.ones((w)).int()*i+1) 
            leader_y = last_center_y[index_new_connection]         
            last_center_y = torch.where(condition, leader_y, y_read)#torch.arange(w, device=device).int())
            last_weight = torch.where(condition, last_weight[index_new_connection], 0)
            last_grad_x = torch.where(condition, last_grad_x[index_new_connection], grad_x)
            last_grad_y = torch.where(condition, last_grad_y[index_new_connection], grad_y)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ЗААААПИСЬ
            x_center[i+1] = last_center_x
            y_center[i+1] = last_center_y
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ЗААААПИСЬ           
            #Склеиваем координаты в массиве
            # merge_coords = torch.concat((last_center_x.unsqueeze(dim=1),last_center_y.unsqueeze(dim=1),last_weight.unsqueeze(dim=1), last_grad_x.unsqueeze(dim=1), last_grad_y.unsqueeze(dim=1)), axis=1)
            t1 = time()
            more_weights, inverse_index = unique_neighbors(last_grad_x, n=6)
            # print(last_weight, more_weights)
            keys = torch.sign(more_weights)
            groop_grad_x = last_grad_x*keys; groop_grad_y = last_grad_y*keys
            groop_weights = last_weight*keys
            #Преобразование групп unique_center, inverse_index, more_weights            
            groop_grad_x = groop_weights*groop_grad_x 
            groop_grad_y = groop_weights*groop_grad_y
            groop_grad_x = groop_grad_x.index_add_(0, inverse_index, last_grad_x)
            groop_grad_y = groop_grad_y.index_add_(0, inverse_index, last_grad_y)
            groop_weights += more_weights
            groop_grad_x /= groop_weights
            groop_grad_y /= groop_weights
            groop_grad_x = groop_grad_x.nan_to_num()
            groop_grad_y = groop_grad_y.nan_to_num()
            r = torch.sqrt(groop_grad_x**2 + groop_grad_y**2)
            groop_grad_x /= r
            groop_grad_y /= r
            groop_grad_x = groop_grad_x.nan_to_num()
            groop_grad_y = groop_grad_y.nan_to_num()
            #Получение значений для массива
            last_grad_x, last_grad_y, last_weight = groop_grad_x[inverse_index], groop_grad_y[inverse_index], groop_weights[inverse_index]
            # print(last_weight, keys)
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ЗААААПИСЬ
            main_grad_x[last_center_x[keys!=0], last_center_y[keys!=0]] = groop_grad_x[keys!=0]
            main_grad_y[last_center_x[keys!=0], last_center_y[keys!=0]] = groop_grad_y[keys!=0]
            weight_of_groops[last_center_x[keys!=0], last_center_y[keys!=0]] = groop_weights[keys!=0]
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!ЗААААПИСЬ
            t4 = time()
            # print(t1-t0,t2-t1,t3-t2,t4-t3)
##################################################################################################
            # if i > 0:
            #     break
    weight_of_groops[-1]=0
    return main_grad_x, main_grad_y, weight_of_groops, x_center, y_center

def contraction(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center,  batch_size):
    # Добавление нижней границы
    shape = main_grad_x.shape
    x_axis = shape[0];
    
    zeros = torch.zeros((x_axis,1), device=x_center.device)
    main_grad_x = torch.cat((main_grad_x,zeros),dim=1);    main_grad_y = torch.cat((main_grad_y,zeros),dim=1)
    x_center = torch.cat((x_center,zeros.int()),dim=1);    y_center = torch.cat((y_center,zeros.int()),dim=1)
    weight_of_groops = torch.cat((weight_of_groops,zeros.int()),dim=1)
    #Добавление нулей
    shape = main_grad_x.shape
    x_axis = shape[0]; y_axis = shape[1]
    x_need = (1+x_axis//batch_size) *batch_size 
    zeros = torch.zeros((x_need - x_axis, shape[1]))
    
    main_grad_x = torch.cat((main_grad_x,zeros));    main_grad_y = torch.cat((main_grad_y,zeros))
    x_center = torch.cat((x_center,zeros.int()));    y_center = torch.cat((y_center,zeros.int()))
    weight_of_groops = torch.cat((weight_of_groops,zeros.int()))
    #Смена индексов
    y_center = y_center + x_center//batch_size * y_axis
    x_center %= batch_size
    #Переформатирование
    main_grad_x = torch.cat(main_grad_x.split(batch_size, dim=0),dim=1);    main_grad_y = torch.cat(main_grad_y.split(batch_size, dim=0),dim=1)
    x_center = torch.cat(x_center.split(batch_size, dim=0),dim=1);     y_center = torch.cat(y_center.split(batch_size, dim=0),dim=1)
    weight_of_groops = torch.cat(weight_of_groops.split(batch_size, dim=0),dim=1)
    return main_grad_x, main_grad_y, weight_of_groops, x_center, y_center, x_axis, x_need, y_axis
          
def extention(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center,   x_axis, x_need, y_axis, batch_size):
    #Возвращение индексов
    y_center = y_center - torch.arange(y_center.shape[1],device=y_center.device)//y_axis*y_axis
    x_center = x_center + torch.arange(y_center.shape[1],device=y_center.device)//y_axis*batch_size
    #Возвращение исходной формы и Убирание лишних нулей
    main_grad_x = torch.cat(main_grad_x.split(y_axis, dim=1),dim=0)[:x_axis,:-1];    main_grad_y = torch.cat(main_grad_y.split(y_axis, dim=1),dim=0)[:x_axis,:-1]
    x_center = torch.cat(x_center.split(y_axis, dim=1),dim=0)[:x_axis,:-1];     y_center = torch.cat(y_center.split(y_axis, dim=1),dim=0)[:x_axis,:-1]
    weight_of_groops = torch.cat(weight_of_groops.split(y_axis, dim=1),dim=0)[:x_axis,:-1]

    return main_grad_x, main_grad_y, weight_of_groops, x_center, y_center
    
def lsd_optimysed(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center, threshold=0.7, batch_size=5):
        # создаём повернутый массив, и меняем х и у местами
    m2x,m2y,w2,y2,x2 = torch.transpose(main_grad_x.clone(),1,0),torch.transpose(main_grad_y.clone(),1,0),torch.transpose(weight_of_groops.clone(),1,0),torch.transpose(x_center.clone(),1,0),torch.transpose(y_center.clone(),1,0)

    m2x,m2y,w2,x2,y2,x2_axis, x2_need, y2_axis =  contraction( m2x,m2y,w2,x2,y2 , batch_size=batch_size)
        # main_grad_x, main_grad_y, weight_of_groops, x_center, y_center = m2x,m2y,w2,x2,y2
    main_grad_x, main_grad_y, weight_of_groops, x_center, y_center, x_axis, x_need, y_axis = contraction(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center,batch_size=batch_size)
    edge = main_grad_x.shape[1]
        #Склеиваем массивы и обрабатываем

    main_grad_x, main_grad_y, weight_of_groops, x_center, y_center = torch.cat((main_grad_x, m2x),dim=1), torch.cat((main_grad_y, m2y),dim=1),torch.cat((weight_of_groops, w2),dim=1),torch.cat((x_center, x2),dim=1),torch.cat((y_center, y2+edge),dim=1),
    main_grad_x, main_grad_y, weight_of_groops, x_center, y_center = one_grad_groop(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center, threshold=threshold)
        # Расзбираем массивы
    m2x,m2y,w2,x2,y2 = main_grad_x.clone()[:,edge:], main_grad_y.clone()[:,edge:], weight_of_groops.clone()[:,edge:], x_center.clone()[:,edge:], y_center.clone()[:,edge:]-edge

    main_grad_x, main_grad_y, weight_of_groops, x_center, y_center = main_grad_x[:,:edge], main_grad_y[:,:edge], weight_of_groops[:,:edge], x_center[:,:edge], y_center[:,:edge]
        # Разархивируем массивы в нормальный вид
    main_grad_x, main_grad_y, weight_of_groops, x_center, y_center = extention(main_grad_x, main_grad_y, weight_of_groops, x_center, y_center,   x_axis, x_need, y_axis, batch_size=batch_size)
    m2x,m2y,w2,x2,y2 =  extention( m2x,m2y,w2,x2,y2,x2_axis, x2_need, y2_axis , batch_size=batch_size)
    y_center = torch.maximum(y_center, torch.zeros(y_center.shape).int())
    y2 = torch.maximum(y2, torch.zeros(y2.shape).int())
    return main_grad_x, main_grad_y, weight_of_groops, x_center, y_center,m2x,m2y,w2,x2,y2


def lsd_meteor_on_image(grad_x, grad_y, threshold=0.9,batch_size=30):
    shape = grad_x.shape;    h,w = grad_x.shape;    device = grad_x.device
    weight_of_groops = torch.ones(shape)
    x_center = (torch.ones(shape, device=device).int() )*torch.arange(h, device=device)[:,None].int()
    y_center = (torch.ones(shape, device=device).int())*torch.arange(w, device=device)[None,:].int()
    hgrad_x, hgrad_y, hweight_of_groops, hx_center, hy_center, m2x,m2y,w2,x2,y2 = lsd_optimysed(grad_x.clone(), grad_y.clone(), weight_of_groops.clone(), x_center.clone(), y_center.clone(), threshold=threshold, batch_size=batch_size)
    # 
    # wgrad_x, wgrad_y, wweight_of_groops, wx_center, wy_center = lsd_optimysed(grad_x.clone(), grad_y.clone(), weight_of_groops.clone(), x_center.clone(), y_center.clone(), threshold=0.7)

    # get_meteors()
    return hweight_of_groops,hx_center, hy_center, hgrad_x, hgrad_y, w2,x2,y2,m2x,m2y