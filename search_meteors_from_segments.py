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
    
    
def meteor_finder(hweight_of_groops,hx_center, hy_center, x_end, y_end, hgrad_x, hgrad_y,threshold_weights, wight_threshold=15, len_comparing=0.4, dop_size = 10,death=5):
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
    compare = torch.where(compare < -0.7, 1, 0)

    #Здесь смотрим точку на оy куда падает линия
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
    compare *= (dy1<wight_threshold)*(dy2<wight_threshold)
    # Выкидываем странные
    teoretical_len = torch.maximum(lex[:,None],rex) - torch.minimum(lx[:,None],rx)
    real_t_len =  torch.minimum(lex[:,None],rex) - torch.maximum(lx[:,None],rx)

    real_t_len = torch.where(real_t_len >0 ,real_t_len, 0)
    
    compare *= real_t_len/teoretical_len > len_comparing
    
    founded_num = compare[compare>0].shape
    #Изъятие значений
        
    #нужно теперь сделать из этого список:
    suitable_arg_list = torch.argwhere(compare > 0)
    #голова и хвост
    p1 = torch.minimum(lex[:,None],rex)
    p2 = torch.maximum(lx[:,None],rx)
    p3 = torch.minimum( torch.minimum(ley[:,None],rey),torch.minimum( ly[:,None], ry))
    p4 =  torch.maximum( torch.maximum(ley[:,None],rey),torch.maximum( ly[:,None], ry))
    coord_rx = lx[:,None]*0 + rx
    coord_rx = coord_rx[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    coord_ry = ly[:,None]*0 + ry
    coord_ry = coord_ry[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    coord_lx = lx[:,None] + rx*0
    coord_lx = coord_lx[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    coord_ly = ly[:,None] + ry*0
    coord_ly = coord_ly[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    add_weight = 20
    
    
    # print(suitable_arg_list.numel())
    if suitable_arg_list.numel() != 0:
        suitable_p1 = p1[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        suitable_p2 = p2[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        suitable_p3 = p3[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        suitable_p4 = p4[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        
        detected_list = []
        death_list =[]
        for i in range(suitable_arg_list.shape[0]):
            # print(i)
            try:
                test = hweight_of_groops[hx_center, hy_center]
                # print(p1[i],p2[i],p3[i],p4[i])
                y0,y1,x0,x1 = suitable_p2[i],suitable_p1[i]+1, -dop_size+suitable_p3[i],suitable_p4[i]+dop_size

                testx = hx_center[y0:y1,x0:x1]
                testy = hy_center[y0:y1,x0:x1]
                test = hweight_of_groops[testx,testy]
                # print(suitable_p2[i],suitable_p1[i], suitable_p3[i],suitable_p4[i])
                # plt.matshow(test.to('cpu'))
                testr = (testx == coord_rx[i]) *( testy == coord_ry[i])
                testl = (testx == coord_lx[i]) *( testy == coord_ly[i])
                testr =torch.where(testr, test,0)
                testl =torch.where(testl, test,0)
                testr *= torch.arange(testr.shape[1],0,-1,device=device)
                testl *= torch.arange(testl.shape[1],device=device)
                # print('tt')
                left_edge = testl.argmax(axis=1)
                right_edge = testr.argmax(axis=1)
                pix_between_edge = right_edge - left_edge 

                enemy_size = torch.mean(pix_between_edge.to(torch.float))

                if enemy_size < death:
                    # print(enemy_size)
                    death_list.append(enemy_size)
                    detected_list.append((suitable_p1[i],suitable_p2[i],suitable_p3[i],suitable_p4[i]))
            except Exception:
                pass
        return len(detected_list), suitable_arg_list.numel(), detected_list
    else:
        return 0, suitable_arg_list.numel(), []
                
        #Теперь надо вытащить координаты центра и сделать массив групп
        
def meteor_segment_finder(hweight_of_groops,hx_center, hy_center, x_end, y_end, hgrad_x, hgrad_y,threshold_weights, wight_threshold=15, len_comparing=0.4, dop_size = 10,death=5, acceptable_crossing=1):
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
    # orientation = torch.atan2( tgx, tgy)

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
    compare = torch.where(compare < -0.7, 1, 0)
    
    # print(compare)

    
    #Здесь смотрим точку на оy куда падает линия
    lk = (ly - ley )/(lx-lex)
    ly0 = ly - lk*lx
    ly0_tens = ly0[:,None] * torch.ones((lx.shape[0],rx.shape[0]), device = device)
    ry0 =  ry - lk[:,None]*rx
    rey0 =  rey - lk[:,None]*rex
    dy1 = ry0 - ly0_tens + acceptable_crossing
    dy2 = rey0 - ly0_tens + acceptable_crossing
    #оставили только правильные пары право лево

    compare *= (dy1>0)*(dy2>0) 
    
    #выкинули далекие
    compare *= (dy1<wight_threshold)*(dy2<wight_threshold)
    # Выкидываем странные
    teoretical_len = torch.maximum(lex[:,None],rex) - torch.minimum(lx[:,None],rx)
    real_t_len =  torch.minimum(lex[:,None],rex) - torch.maximum(lx[:,None],rx)

    real_t_len = torch.where(real_t_len >0 ,real_t_len, 0)
    
    compare *= real_t_len/teoretical_len > len_comparing
    
    # plt.matshow(compare.to('cpu'))
    
    founded_num = compare[compare>0].shape
    #Изъятие значений
        
    #нужно теперь сделать из этого список:
    suitable_arg_list = torch.argwhere(compare > 0)
    #голова и хвост
    p1 = torch.minimum(lex[:,None],rex)
    p2 = torch.maximum(lx[:,None],rx)
    p3 = torch.minimum( torch.minimum(ley[:,None],rey),torch.minimum( ly[:,None], ry))
    p4 =  torch.maximum( torch.maximum(ley[:,None],rey),torch.maximum( ly[:,None], ry))
    coord_rx = lx[:,None]*0 + rx
    coord_rx = coord_rx[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    coord_ry = ly[:,None]*0 + ry
    coord_ry = coord_ry[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    coord_lx = lx[:,None] + rx*0
    coord_lx = coord_lx[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    coord_ly = ly[:,None] + ry*0
    coord_ly = coord_ly[suitable_arg_list[:,0],suitable_arg_list[:,1]]
    add_weight = 20
    
    if suitable_arg_list.numel() != 0:
        suitable_p1 = p1[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        suitable_p2 = p2[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        suitable_p3 = p3[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        suitable_p4 = p4[suitable_arg_list[:,0],suitable_arg_list[:,1]]
        detected_list = []
        death_list =[]
        for i in range(suitable_arg_list.shape[0]):
            # print(i)
            try:
                test = hweight_of_groops[hx_center, hy_center]
                # print(p1[i],p2[i],p3[i],p4[i])
                y0,y1,x0,x1 = suitable_p2[i],suitable_p1[i]+1, -dop_size+suitable_p3[i],suitable_p4[i]+dop_size

                testx = hx_center[y0:y1,x0:x1]
                testy = hy_center[y0:y1,x0:x1]
                test = hweight_of_groops[testx,testy]
                # print(suitable_p2[i],suitable_p1[i], suitable_p3[i],suitable_p4[i])
                # plt.matshow(test.to('cpu'))
                testr = (testx == coord_rx[i]) *( testy == coord_ry[i])
                testl = (testx == coord_lx[i]) *( testy == coord_ly[i])
                testr =torch.where(testr, test,0)
                testl =torch.where(testl, test,0)
                testr *= torch.arange(testr.shape[1],0,-1,device=device)
                testl *= torch.arange(testl.shape[1],device=device)
                # print('tt')
                left_edge = testl.argmax(axis=1)
                right_edge = testr.argmax(axis=1)
                pix_between_edge = right_edge - left_edge
                enemy_size = torch.mean(pix_between_edge.to(torch.float))

                if enemy_size < death:
                    center_line = (right_edge + left_edge)/2. 
                    mknx = torch.arange(center_line.shape[0], device=device)
                    k = center_line.shape[0]* torch.sum(mknx*center_line) - torch.sum(mknx)*torch.sum(center_line)
                    k /= center_line.shape[0]*torch.sum(mknx**2) - torch.sum(mknx)**2
                    b = (torch.sum(center_line) - k*torch.sum(mknx) )/center_line.shape[0]

                    death_list.append(enemy_size)
                    # print(enemy_size)
                    detected_list.append(torch.stack((y0,y1,x0+b,x0+b+k*(y1-y0))))#+b,x0+b+k*(x1-x0))))#(suitable_p1[i],suitable_p2[i],suitable_p3[i],suitable_p4[i])) )
                    # detected_list.append(torch.stack((suitable_p1[i],suitable_p2[i],suitable_p3[i],suitable_p4[i])) )

            except Exception:
                pass
        return len(detected_list), suitable_arg_list.numel(), detected_list
    else:
        return 0, suitable_arg_list.numel(), []
                
   