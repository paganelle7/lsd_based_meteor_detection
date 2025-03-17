import torch

def endge_comparing(hweight_of_groops,hx_center, hy_center, x_end, y_end, hgrad_x, hgrad_y,threshold_weights,wight_threshold=10, len_comparing=0.7):
    device=hweight_of_groops.device
    tt = torch.where(hweight_of_groops>threshold_weights)
    tx = hx_center[hweight_of_groops>threshold_weights]
    ty = hy_center[hweight_of_groops>threshold_weights]
    tex = x_end[hweight_of_groops>threshold_weights]
    tey = y_end[hweight_of_groops>threshold_weights]
    tgx = hgrad_x[hweight_of_groops>threshold_weights]
    tgy = hgrad_y[hweight_of_groops>threshold_weights]
    
    #делим на левые и правые
    orientation = torch.atan2( tgy, tgx)
    lx = tx[orientation>0]
    lex = tex[orientation>0]
    ly = ty[orientation>0]
    ley = tey[orientation>0]
    lgx = tgx[orientation>0]
    lgy = tgy[orientation>0]
        
    rx = tx[orientation<0]
    rex = tex[orientation<0]
    ry = ty[orientation<0]
    rey = tey[orientation<0]
    rgx = tgx[orientation<0]
    rgy = tgy[orientation<0]
    #Выкинули разные градиенты
    compare = torch.zeros((lx.shape[0],rx.shape[0]), device = device)
    compare += lgx[:,None]*rgx + lgy[:,None]*rgy
    compare = torch.where(compare < -0.8, 1, 0)
    
    lk = (ly - ley )/(lx-lex)
    ly0 = ly - lk*lx
    ly0_tens = ly0[:,None] * torch.ones((lx.shape[0],rx.shape[0]), device = device)
    ry0 =  ry - lk[:,None]*rx
    rey0 =  rey - lk[:,None]*rex
    dy1 = ry0 - ly0_tens
    dy2 = rey0 - ly0_tens
    #оставили только правильные пары право лево
    compare *= (dy1>0)*(dy2>0) 
    #выкинули далекие
    # wight_threshold = 10
    compare *= (dy1<wight_threshold)*(dy2<wight_threshold)
    # Выкидываем странные
    teoretical_len = torch.maximum(lex[:,None],rex) - torch.minimum(lx[:,None],rx)
    real_t_len =  torch.minimum(lex[:,None],rex) - torch.maximum(lx[:,None],rx)
    real_t_len = torch.where(real_t_len >0 ,real_t_len, 0)
    
    
    compare *= real_t_len/teoretical_len > len_comparing
    
    founded_num = compare[compare>0].shape
    #Изъятие значений
    
    if compare.numel() > 0 :
    
        r_ret = compare.max(axis=0)[0]
        l_ret = compare.max(axis=1)[0]

        layout =  lx[l_ret>0], ly[l_ret>0], lex[l_ret>0], ley[l_ret>0]
        rayout =  rx[r_ret>0], ry[r_ret>0], rex[r_ret>0], rey[r_ret>0]
        # print(l_ret)


        # plt.matshow(compare.to('cpu'),cmap='YlGnBu')
        # plt.colorbar()
        val,_ = torch.max(compare, axis=1)
        argval = torch.where(val!=0)

        return  founded_num, layout, rayout #lx[l_ret>0], ly[l_ret>0], lex[l_ret>0], ley[l_ret>0]#rex,rey, r_out*0,r_out# rx,ry#tx[argval], ty[argval]
    else:
        return 0, None, None