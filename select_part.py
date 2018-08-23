# -*- coding: utf-8 -*-
import cv2
import pygame  
from PIL import Image  
import random  
import math
from PIL import ImageEnhance
import sys  
from pygame.locals import *  
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "img",
    help="aaa",
    type=str
)

args = parser.parse_args()
file4 = args.img


filename = file4
img_origin = cv2.imread(file4)
size =width,height= int(img_origin.shape[1]/4),int(img_origin.shape[0]/4)
#size=width,height=1000,750
#size_1=width_1,height_1=1300,976
turtle_ori=pygame.image.load(filename)

turtle = pygame.transform.scale(turtle_ori, (size)) 


def rotate_bound(image, pos_1,pos_2):  
    # grab the dimensions of the image and then determine the  
    # center  
    (h, w) = image.shape[:2]
    if pos_1[0]-pos_2[0]!=0:
        tan = abs((pos_1[1]-pos_2[1])/(pos_1[0]-pos_2[0]))
    else:
        tan = 9999999
    rot=0

    if tan>=1:
        if pos_1[1]>pos_2[1]:
            rot=180
            pos_1_rot=(w-pos_1[0]*4,h-pos_1[1]*4)
            pos_2_rot=(w-pos_2[0]*4,h-pos_2[1]*4)
        else:
            rot=0
    else:

        if pos_1[0]<pos_2[0]:
            rot= 270
        
            pos_1_rot=(h-pos_1[1]*4,pos_1[0]*4)
            pos_2_rot=(h-pos_2[1]*4,pos_2[0]*4)
        else:
            rot=90
            pos_1_rot=(pos_1[1]*4, w-pos_1[0]*4)
            pos_2_rot=(pos_2[1]*4, w-pos_2[0]*4)
    print(rot)
      
    (cX, cY) = (w // 2, h // 2)  
  
    # grab the rotation matrix (applying the negative of the  
    # angle to rotate clockwise), then grab the sine and cosine  
    # (i.e., the rotation components of the matrix)  
    M = cv2.getRotationMatrix2D((cX, cY), rot, 1.0)  
    cos = np.abs(M[0, 0])  
    sin = np.abs(M[0, 1])  
  
    # compute the new bounding dimensions of the image  
    nW = int((h * sin) + (w * cos))  
    nH = int((h * cos) + (w * sin))  
  
    # adjust the rotation matrix to take into account translation  
    M[0, 2] += (nW / 2) - cX  
    M[1, 2] += (nH / 2) - cY  
  
    # perform the actual rotation and return the image  
    return cv2.warpAffine(image, M, (nW, nH)),pos_1_rot,pos_2_rot

def select_grid_area(filename, step):

    pygame.init()   
    
      
    bg=(255,255,255)  
    #clock=pygame.time.Clock()  
    screen=pygame.display.set_mode((width + 250,height), pygame.RESIZABLE)  
      
      
    pygame.display.set_caption('Lizard tool') 
    
  
    position=turtle.get_rect()  
    position.center = width // 2, height // 2  
 
    select=0   
    select_rect=pygame.Rect(0,0,0,0)  

    drag=0  
    flag = True  
    while flag:  
        for event in pygame.event.get():  
            if event.type==QUIT:  
                sys.exit()  
            elif event.type==MOUSEBUTTONDOWN:  
                if event.button==1:  
                    if select==0 and drag==0:  
                        pos_start = event.pos  
                        select=1  
                    elif select==2 and drag==0:  
                        capture=screen.subsurface(select_rect).copy()  
                        cap_rect=capture.get_rect()
                        print ('________________________'+str(cap_rect))
                        drag=1  

                    elif select==2 and drag==2:  
                        flag = False
      
            elif event.type==MOUSEBUTTONUP:  
                if event.button==1:  
                    if select==1 and drag==0:  
                        pos_stop=event.pos  
                        select=2  
                          
                    if select==2 and drag==1:  
                        drag=2                        
                                 
        screen.fill(bg)  
        screen.blit(turtle,position)    
        cur_font = pygame.font.SysFont("arial", 35)  
        text = 'LIZARD TOOLS'
        text_fmt = cur_font.render(text, 1, (255,0,0))  
        screen.blit(text_fmt, (width,0,200,50))
        cur_font = pygame.font.SysFont("arial", 25)  
        text = 'Step1:select grid area'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (width,50,200,50))
        text = 'Step2:select head/body point'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (width,80,200,50))
        
        text = 'This is step:'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (width,160,200,50))
        
        cur_font = pygame.font.SysFont("arial", 120)
        text = str(step)
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (width,210,200,50))
        if step == 2:
            cur_font = pygame.font.SysFont("arial", 25)  
            text = 'pixels of one grid(1mm):'
            text_fmt = cur_font.render(text, 1, (0,0,0))
            screen.blit(text_fmt, (width,300,200,50))
            cur_font = pygame.font.SysFont("arial", 30)  
            text = str(grid_length)
            text_fmt = cur_font.render(text, 1, (0,0,0))
            screen.blit(text_fmt, (width,330,200,50))
        if  select:  
            mouse_pos=pygame.mouse.get_pos()  
            if select==1:  
                pos_stop=mouse_pos  
            select_rect.left, select_rect.top = pos_start  
            select_rect.width, select_rect.height = pos_stop[0] - pos_start[0], pos_stop[1] - pos_start[1]  
            pygame.draw.rect(screen,(0,0,0),select_rect,1)
        if drag:  
            if drag==1:  
               cap_rect.center=mouse_pos  
            screen.blit(capture,cap_rect)  
          
        pygame.display.flip()
    print ('select_rect:'+str(select_rect.left)+ str(select_rect.top)+ str(select_rect.height)+ str(select_rect.width))
    pygame.quit()
    #exit()
#    print ('turtle'+str(turtle_ori))
    return select_rect

def process_grid(filename, select_rect):
    image = Image.open(filename)
    ori_width ,ori_height = image.size
    ratio = float(ori_height)/float(height)
    p1 = select_rect.left * ratio
    p2 = select_rect.top * ratio
    p3 = select_rect.width * ratio + p1
    p4 = select_rect.height * ratio + p2
#    print (p1+p2+p3+p4)
    img = image.crop((p1, p2, p3, p4))
    img = ImageEnhance.Contrast(img).enhance(1.5)  
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img.save('enhanced.jpg')
    
    
    img = cv2.imread('enhanced.jpg')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    HT = 2000
    flat = True
    
    kp_num_min = 50
    kp_num_max = 300
    while flat:
        #print HT
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold = HT,nOctaveLayers = 1)
        kp, des = surf.detectAndCompute(gray_img, None)
        #print len(kp)
        if kp_num_min < len(kp) < kp_num_max:
            flat = False
        elif len(kp) < kp_num_min:
            HT -= 500
        elif len(kp) > kp_num_max:
            HT += 500
#    print (len(kp))
    img = cv2.drawKeypoints(gray_img, kp, img)
    # add keyPoint position to a list
    kp_position = []
    
    for i in range(len(kp)):
        kp_position.append(kp[i].pt)
    #print kp_position
    #print des[2], des[3], des[2] < 0.6*des[3]
    
    cv2.imwrite('surf.jpeg',img)
    
    #random to caculate the neighbours of a KeyPoint
    ITER = 300
    all_grid_points = []
    for _ in range(ITER):
        # the keypoint that can make a grid with the random point
        len_min = 75
        len_max = 90
        grid_point = [] 
        possible_point = []
        random_point = random.randint(0,len(kp_position)-1)
        x1,y1 = kp_position[random_point]
        #print x1,y1
        for i in range(len(kp_position)):
            x2,y2 = kp_position[i]
            # draw a circle add the within keypoint
            if 60 < math.sqrt((x1 - x2)**2 + (y1 - y2)**2) < 140: 
                #print ('length',math.sqrt((x1 - x2)**2 + (y1 - y2)**2))
                possible_point.append((x2,y2))
        for i in range(len(possible_point)):
            x2,y2 = possible_point[i]
            l1 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            for j in range(i,len(possible_point)):
                x3,y3 = possible_point[j]
                l2 = math.sqrt((x1 - x3)**2 + (y1 - y3)**2)
                l3 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2)
                if abs(l1 - l2) < (l1 + l2)/200 and abs(l3 - math.sqrt(2)*(l1 + l2)/2) < (l1+l2)/200:
                    #print l1,l2
                    if len_min <(l1 + l2) / 2 < len_max:
                        grid_point.append((l1 + l2) / 2)
                elif abs(l1 - l3) < (l1 + l3)/200 and abs(l2 - math.sqrt(2)*(l1 + l3)/2) < (l1+l3)/200:
                    #print l1,l3
                    if len_min <(l1 + l3) / 2 < len_max:
                        grid_point.append((l1 + l3) / 2)
                elif abs(l2 - l3) < (l3 + l2)/200 and abs(l1 - math.sqrt(2)*(l2 + l3)/2) < (l2+l3)/200:
                    #print l3,l2 
                    if len_min <(l3 + l2) / 2 < len_max:
                        grid_point.append((l3 + l2) / 2)
#        if grid_point != []:
#            print (grid_point)
        all_grid_points += grid_point
    #print all_grid_points
    if len(all_grid_points) == 0:
        print ('--------------please try another grid area----------------')
        sys.exit()
    return sum(all_grid_points)/len(all_grid_points)

def process_contours(filename):
    img = cv2.imread(filename)
#    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    top = 16  # shape[0] = rows
    bottom = 16
    left = 16 # shape[1] = cols
    right = 16
#        50
    value = [255,255,255]
    borderType = cv2.BORDER_CONSTANT
    img_bordered = cv2.copyMakeBorder(img, top, bottom, left, right, borderType, None, value)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img,(3,3),0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,203,3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    k = np.ones((3,3),np.uint8)
    opened = cv2.erode(opened, k, iterations =4)
    opened = cv2.dilate(opened, k, iterations = 4)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    close = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    k = np.ones((8,8),np.uint8)
    close = cv2.erode(close, k, iterations =4)
    close = cv2.dilate(close, k, iterations = 4)
    
    top = 16  
    bottom = 16
    left = 16 
    right = 16
        
    value = 255
    borderType = cv2.BORDER_CONSTANT
    close = cv2.copyMakeBorder(close, top, bottom, left, right, borderType, None, value)

        
    (im2,contours, _) = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    maxmum = 0
    maxid=0
    for i in range(1,len(contours)):
        if contours[i].shape[0]>maxmum:
            maxmum = contours[i].shape[0]
            maxid = i

    cont = contours[maxid]
    filterRadius = 200
    filterSize = 2 * filterRadius + 1
    sigma = 40
    length_ = cont.shape[0] + 2 * filterRadius
    idx = (cont.shape[0] - filterRadius)
    x=np.array([])
    y=np.array([])
    for i in range(length_):
        x=np.append(x,cont[(idx+i)%cont.shape[0]][0][0])
        y=np.append(y,cont[(idx+i)%cont.shape[0]][0][1])
    x_fix = cv2.GaussianBlur(x,(filterSize ,filterSize ),sigma,sigma)
    y_fix = cv2.GaussianBlur(y,(filterSize ,filterSize ),sigma,sigma)
    smooth_cont = np.array([[[0,0]]])
    for i in range(filterRadius,cont.shape[0]+filterRadius):
        if int(x_fix[i][0])!=smooth_cont[-1][0][0] or int(y_fix[i][0])!=smooth_cont[-1][0][1]:
            smooth_cont = np.insert(smooth_cont,smooth_cont.shape[0],[[[x_fix[i][0],y_fix[i][0]]]],axis=0)
    
    smooth_cont = np.delete(smooth_cont,0,0)
    cv2.drawContours(img_bordered, [smooth_cont], -1, (0, 255, 0), 5)
    return img_bordered, smooth_cont

def select_feature_point():
    pygame.init()   
    bg=(255,255,255)  
    #clock=pygame.time.Clock()  
    screen=pygame.display.set_mode((size_1[0] + 250,size_1[1]),pygame.RESIZABLE)      
    pygame.display.set_caption('Lizard tool') 

    position=turtle.get_rect()  
    position.center = size_1[0] // 2, size_1[1] // 2  
    select=0    
    drag=0  
    flag = True  
    while flag:  
        for event in pygame.event.get():  
            if event.type==QUIT:  
                sys.exit()  
            elif event.type==MOUSEBUTTONDOWN:  
                if event.button==1:  
                    if select==0:
                        
                        select = 1
                        pos_1 = event.pos
                    elif select==1: 
                        pos_2 = event.pos
                        flag=False
                        select =2
                       
        screen.fill(bg)  
        screen.blit(turtle,position)   
        cur_font = pygame.font.SysFont("arial", 35)  
        text = 'LIZARD TOOLS'
        text_fmt = cur_font.render(text, 1, (255,0,0))  
        screen.blit(text_fmt, (size_1[0],0,200,50))
        cur_font = pygame.font.SysFont("arial", 25)  
        text = 'Step1:select grid area'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_1[0],50,200,50))
        
        text = 'Step2:select head/body point'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_1[0],80,200,50))
        text = 'This is step:'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_1[0],160,200,50))
        
        cur_font = pygame.font.SysFont("arial", 120)
        text = '2'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_1[0],190,200,50))
#        if step == 2:
        cur_font = pygame.font.SysFont("arial", 25)  
        text = 'pixels of one grid(1mm):'
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_1[0],300,200,50))
        cur_font = pygame.font.SysFont("arial", 30)  
        text = str(grid_length)
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_1[0],330,200,50))
        if  select==1:  
#            pygame.draw.rect(screen,(0,0,0),select_rect,1)
            pygame.draw.circle(screen, (0,255,0), pos_1, 5)
#            pygame.draw.circle(screen, (0,255,0), pos_2, 5)
        elif select==2:
            pygame.draw.circle(screen, (0,255,0), pos_1, 5)
            pygame.draw.circle(screen, (0,255,0), pos_2, 5)
          
        pygame.display.flip()
#    print ('select_rect:'+str(select_rect.left)+ str(select_rect.top)+ str(select_rect.height)+ str(select_rect.width))
    pygame.quit()
#    #exit()
#    print ('turtle'+str(turtle_ori))
    return pos_1, pos_2

def measure_length(contours_img,contours,pos_1,pos_2):
    cont = contours
    pt_h = (pos_1[0]+16,pos_1[1]+16)
    pt = (pos_2[0]+16,pos_2[1]+16)
    print(pt_h)
    print(pt)
    maxdst = 0
    min_len=999999
    min_i = cont[0]
    for i in range(cont.shape[0]):
        
        length = (cont[i][0][0]-pt[0])**2+(cont[i][0][1]-pt[1])**2
        if min_len>length:
            min_len = length
            min_i=cont[i]
            
    
    if (min_i[0][1]-pt[1])!=0:
        k_ = -(min_i[0][0]-pt[0])/(min_i[0][1]-pt[1])
        up_x = pt[0]-50/((k_**2+1)**0.5)
        up_y = pt[0]-50*k_/((k_**2+1)**0.5)
        down_x = pt[0]+50/((k_**2+1)**0.5)
        down_y = pt[0]+50*k_/((k_**2+1)**0.5)
    else:
        up_x =pt[0]
        up_y=pt[1]-50
        down_x =pt[0]
        down_y=pt[1]+50
    
    min_len_up=999999
    for i in range(cont.shape[0]):
        
        length = (cont[i][0][0]-up_x)**2+(cont[i][0][1]-up_y)**2
        if min_len_up>length:
            min_len_up = length
            min_i_up=cont[i]
    min_len_down=999999
    for i in range(cont.shape[0]):
        
        length = (cont[i][0][0]-down_x)**2+(cont[i][0][1]-down_y)**2
        if min_len_down>length:
            min_len_down = length
            min_i_down=cont[i]
    headpt=()
    for c in cont:
        dst_up=(c[0][0]-pt_h[0])**2+(c[0][1]-pt_h[1])**2
        dst_down=(c[0][0]-pt[0])**2+(c[0][1]-pt[1])**2
        if dst_up<=dst_down:
            dst = (c[0][0]-pt[0])**2 +(c[0][1]-pt[1])**2
            if maxdst < dst:
                maxdst=dst
                headpt = (c[0][0],c[0][1])
    head_edge=[]
    if (headpt[1]-pt_h[1])!=0:
        k_h = -(headpt[0]-pt_h[0])/(headpt[1]-pt_h[1])
        for c in cont:
            dst_up=(c[0][0]-pt_h[0])**2+(c[0][1]-pt_h[1])**2
            dst_down=(c[0][0]-pt[0])**2+(c[0][1]-pt[1])**2
            if dst_up<dst_down:
                if c[0][0]-pt_h[0]!=0:
                    k_c = (c[0][1]-pt_h[1])/(c[0][0]-pt_h[0])
    #                if k_c/k_h>0.99 and k_c/k_h<1.01:
    #                print(k_c)
                    
                    if abs(math.atan(k_c)-math.atan(k_h))<0.005:
                        head_edge.append(c)
    #            else:
    #                k_c=99999999
    head_len_pt=[]
    head_one = head_edge[0]
    head_two = head_edge[-1]
    for i in range(1,len(head_edge)):
        dst = (head_edge[i][0][0]-head_edge[0][0][0])**2+(head_edge[i][0][1]-head_edge[0][0][1])**2
        if dst>20:
            head_two = head_edge[i]
            break
    #sim_set=[]
    head_len_pt.append(head_one)
    head_len_pt.append(head_two)
    
    p=[0.1,0.2,0.3,0.4,0.5,0.6,0.65,0.75,0.85]
    total_len=abs(pt[1]-headpt[1])
    x_len = abs(pt[0]-headpt[0])
    pts = []
#    last_x=0
    for i in range(9):
        y_ = int(headpt[1]+total_len*p[i])
        x_set = []
        for j in range(cont.shape[0]):
            if cont[j][0][1]==y_ and cont[j-1][0][1]!=y_:
                x_set.append(cont[j][0])
        x_center = headpt[0]+x_len*p[i]
        x_dst=[]
        if len(x_set)==2:
            x_ = int((x_set[0][0]+x_set[1][0])/2)
        elif len(x_set)>2:
            for k in range(len(x_set)):
                dst = abs(x_set[k][0]-x_center)
                x_dst.append(dst)
            x_1 = 0
            x_2 = 0
            x_max = 0
            x_max_2 = 0
            for k in range(len(x_set)):
                if x_dst[k]>x_max:
                    x_max_2 = x_max
                    x_max = x_dst[k]
                    x_2=x_1
                    x_1=k
            x_ = int((x_set[x_1][0]+x_set[x_2][0])/2)
    #    if i>1:
    #        x_ = int((x_+last_x)/2)
    #    last_x = x_
        pts.append((x_,y_))
            
    
    lineThickness = 10
    cv2.line(contours_img, headpt, pts[0], (0,0,255), lineThickness)
    cv2.line(contours_img, pts[0], pts[1], (0,0,255), lineThickness)
    cv2.line(contours_img, pts[1], pts[6], (0,0,255), lineThickness)
    cv2.line(contours_img, pts[6], pts[7], (0,0,255), lineThickness)
    cv2.line(contours_img, pts[7], pt, (0,0,255), lineThickness)
    
    pt_1 = (head_len_pt[0][0][0],head_len_pt[0][0][1])
    pt_2 = (head_len_pt[1][0][0],head_len_pt[1][0][1])
    cv2.line(contours_img, pt_1, pt_2, (255,0,0), lineThickness)
    
    cv2.imwrite("line.jpg",contours_img)
    head_dst = ((pt_1[0]-pt_2[0])**2+(pt_1[1]-pt_2[1])**2)**0.5
    body_dst = ((headpt[0]-pts[0][0])**2+(headpt[1]-pts[0][1])**2)**0.5
    #body_dst+= ((pts[0][0]-pts[1][0])**2+(pts[0][1]-pts[1][1])**2)**0.5
    body_dst+= ((pts[1][0]-pts[6][0])**2+(pts[1][1]-pts[6][1])**2)**0.5
    body_dst+= ((pts[6][0]-pts[7][0])**2+(pts[6][1]-pts[7][1])**2)**0.5
    body_dst+= ((pts[7][0]-pt[0])**2+(pts[7][1]-pt[1])**2)**0.5
    head_length = head_dst/grid_length
    body_length = body_dst/grid_length
    return  head_length, body_length

def final_step(head_length, body_length):
    pygame.init()   
    #screen=pygame.display.set_mode((width ,height), pygame.RESIZABLE)  
    
    bg=(255,255,255)  
        #clock=pygame.time.Clock()  
    screen=pygame.display.set_mode((size_2[0] + 250,size_2[1]), pygame.RESIZABLE)  
    pygame.display.set_caption('Lizard tool') 
    position=turtle.get_rect()  
    position.center = size_2[0] // 2, size_2[1] // 2  
    flag = True  #
    while flag:  
        for event in pygame.event.get():  
            if event.type==QUIT:  
                sys.exit()  
            elif event.type==MOUSEBUTTONDOWN:  
                if event.button==1:  
                    flag =False
        screen.fill(bg)  
        screen.blit(turtle,position)  
        cur_font = pygame.font.SysFont("arial", 35)  
        text = 'LIZARD TOOLS'
        text_fmt = cur_font.render(text, 1, (255,0,0))  
        screen.blit(text_fmt, (size_2[0],0,200,50))
        cur_font = pygame.font.SysFont("arial", 25)  
        text = 'Step1:select grid area'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_2[0],50,200,50))
        
        text = 'Step2:select head/body point'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_2[0],80,200,50))
        
        text = 'This is step:'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_2[0],160,200,50))
        
        cur_font = pygame.font.SysFont("arial", 120)
        text = '3'
        text_fmt = cur_font.render(text, 1, (0,0,255))
        screen.blit(text_fmt, (size_2[0],190,200,50))
        #        if step == 2:
        cur_font = pygame.font.SysFont("arial", 25)  
        text = 'pixels of one grid(1mm):'
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_2[0],300,200,50))
        text = 'head length is:'
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_2[0],400,200,50))
        text = str(head_length)
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_2[0],450,200,50))
        text = 'body length is:'
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_2[0],500,200,50))
        text = str(body_length)
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_2[0],550,200,50))
        # 显示单位表格长度
        cur_font = pygame.font.SysFont("arial", 30)  
        text = str(grid_length)
        text_fmt = cur_font.render(text, 1, (0,0,0))
        screen.blit(text_fmt, (size_2[0],330,200,50))
        pygame.display.flip()
    pygame.quit()
    


select_rect_grid = select_grid_area(file4, 1)
grid_length = process_grid(file4, select_rect_grid)
#contours_img,contours = process_contours(file4)
#contours_img.shape
#cv2.imwrite('contours_img.jpg', contours_img)
turtle_ori=pygame.image.load(file4)
#pygame.image.save(turtle_ori,'ppp.jpg')
img_origin = cv2.imread(file4)
size_1 = int(img_origin.shape[1]/4),int(img_origin.shape[0]/4)
turtle = pygame.transform.scale(turtle_ori, (size_1))
pos_1,pos_2= select_feature_point()
print(pos_1)

img_rotate,pos_1_rot, pos_2_rot = rotate_bound(img_origin,pos_1,pos_2)
cv2.imwrite('img_rotate.jpg', img_rotate)
contours_img,contours = process_contours('img_rotate.jpg')
#cv2.imwrite('contours_img.jpg', contours_img)
#turtle_ori=pygame.image.load('contours_img.jpg')
#size_2 = int(contours_img.shape[1]/4),int(contours_img.shape[0]/4)
#turtle = pygame.transform.scale(turtle_ori, (size_1))

head_length, body_length = measure_length(contours_img,contours,pos_1_rot,pos_2_rot)

turtle_ori=pygame.image.load('line.jpg')
size_2 = int(contours_img.shape[1]/4),int(contours_img.shape[0]/4)
turtle = pygame.transform.scale(turtle_ori, (size_2))

final_step(head_length, body_length)



#select_rect_lizard = select_grid_area(file4, 3)

