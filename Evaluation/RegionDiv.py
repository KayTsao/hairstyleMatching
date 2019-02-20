import os
import sys
import shutil
import os
import sys
import glob
import shutil
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
offset=981


def Draw_Region(i_dir,o_dir,brow, nose, chin):
    img = Image.open(i_dir)#读取照片
    draw = ImageDraw.Draw(img)
    draw.line([(981+291, brow),(981+1108,brow)],fill=(255,0,0))
    draw.line([(981+291, nose),(981+1108,nose)],fill=(255,0,0))
    draw.line([(981+291, chin),(981+1108,chin)],fill=(255,0,0))
    i = 1
    j = 1
    width = img.size[0]#长度
    height = img.size[1]#宽度
    Area1=0
    Area2=0
    Area3=0
    Area4=0
    #Mask Region
    for i in range(offset+291,offset+1108):#遍历有限宽度的点
        for j in range(230,1047):#遍历有限高度的点
            data = (img.getpixel((i,j)))#获取该点像素值
            if(data[0] == 0 and data[1] == 0 and data[2] > 100):
                img.putpixel((i,j),(255,255,255,0))
            else:
                if(j<brow):
                    Area1 += 1
                elif(j>brow and j<nose):
                    Area2 += 1
                elif(j>nose and j<chin):
                    Area3 += 1
                elif(j>chin):
                    Area4 += 1
    hairArea = Area1 + Area2 + Area3 + Area4
    print(hairArea, Area1/hairArea ,Area2/hairArea , Area3/hairArea , Area4/hairArea)
    img.show()    
    return img


def change_color(i_dir,o_dir):
    i = 1
    j = 1
    img = Image.open(i_dir)#读取照片
    width = img.size[0]#长度
    height = img.size[1]#宽度

#Image Region
    for i in range(offset+291,offset+1108):#遍历有限宽度的点
        for j in range(230,1047):#遍历有限高度的点
            data = (img.getpixel((i,j)))#获取该点像素值
            #if(j==231):
            #    print(data[0],data[1],data[2])
            if(data[0] == 0 and data[1] == 0 and data[2] > 100):
                img.putpixel((i,j),(255,255,255,0))
    img.show()
    return img
    

'''
            data = (img.getpixel((i,j)))#获取该点像素值
            if (data[0] == 0 and data[1] == 0):#Blue Area
                img.putpixel((i,j),(0,0,0,125))#则这些像素点的颜色改成白色
            else:
                img.putpixel((i,j),(100,0,0,75))#则这些像素点的颜色改成透明红色
            img_data = (img.getpixel((i-offset,j)))
            msk_data = (img.getpixel((i,j)))
            img.putpixel((i,j),(img_data[0]+msk_data[0], img_data[1]+msk_data[1], img_data[2]+msk_data[2], 255))
''' 
def region_test(i_dir):
    img = Image.open(i_dir)#读取照片
    width = img.size[0]#长度
    height = img.size[1]#宽度

    for i in range(0,width):#遍历有限宽度的点
        for j in range(0,height):#遍历有限高度的点
            if i==293:
                data=img.getpixel((i,j))
                if(data[0]==255 and data[1] ==255 and data[2]==255):
                    print(i,j)
               


def parse_command_line():
    #if len(sys.argv) < 2:
    #    display_help()
    #    exit(-1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    return input_dir, output_dir


def parse_command_line2():    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    brow = sys.argv[3]
    nose = sys.argv[4]
    chin = sys.argv[5]
    return input_dir, output_dir, brow, nose, chin

if __name__ == '__main__':
    if(len(sys.argv) == 3):
        i_dir, o_dir = parse_command_line()
    elif(len(sys.argv) == 6):
        i_dir, o_dir, brow, nose, chin= parse_command_line2()
    else:
        exit(-1)

    path = i_dir
    #files = os.listdir(path)
    #total_num = len(files)
    #s = []
    #idx=0
    #region_test(i_dir)
    #img = change_color(i_dir, o_dir)#[0,255,0,255])
    img = Draw_Region(i_dir, o_dir, float(brow), float(nose), float(chin))#[0,255,0,255])
   
'''
    for f in files:
        idx +=1 
        src_path = i_dir + f
        dst_path = o_dir + f
        img = change_color(src_path, dst_path)#[0,255,0,255])
        break   
        print('%i of %i' %(idx, total_num))
'''


#img = img.convert("RGB")#把图片强制转成RGB
#img.save(o_dir)#保存修改像素点后的图片
