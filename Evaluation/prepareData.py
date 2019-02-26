import os
import sys
import shutil
import os
import sys
import glob
import shutil
from PIL import Image
import matplotlib.pyplot as plt
def change_color(i_dir,o_dir):
    i = 1
    j = 1
    img = Image.open(i_dir)#读取照片
    width = img.size[0]#长度
    height = img.size[1]#宽度
    offset=450
    #offset2=438
    for i in range(439,720):#遍历有限宽度的点
        for j in range(160,441):#遍历有限高度的点
            img.putpixel((i,j),(255,255,0,255))
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
    
def png2jpg(i_dir, o_dir):
    files = os.listdir(i_dir)
    total_num = len(files)
    print(total_num)
    idx=1
    for f in files:
        print('%i of %i' %(idx, total_num))
        portion = os.path.splitext(f)
        if portion[1] == ".png":
            img = Image.open(i_dir + f)
            newname = portion[0] + '.jpg'
            img.save(newname)
        #os.rename(os.path.join(i_dir,f),os.path.join(i_dir,newname))
        idx += 1

def parse_command_line():
    if len(sys.argv) < 2:
        display_help()
        exit(-1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    return input_dir, output_dir

if __name__ == '__main__':
    i_dir, o_dir = parse_command_line()
    png2jpg(i_dir, o_dir)
'''
    size = 224,224
    files = os.listdir(i_dir)
    total_num = len(files)
    print(total_num)
    idx=1
    
    for f in files:
        print('%i of %i' %(idx, total_num))
        img = Image.open(i_dir + f)
        img = img.convert("L")
        img = img.resize((size), Image.ANTIALIAS)
        
        f_Id = str(idx)
        f_Id = f_Id.zfill(3)
        newname = 'pt-' + f_Id + '.png'
        newname = os.path.join(o_dir,newname)
        print(newname)
        img.save(newname)
        #os.rename(os.path.join(i_dir,f),os.path.join(i_dir,newname))
        idx += 1
 '''      
#img = img.convert("RGB")#把图片强制转成RGB
#img.save(o_dir)#保存修改像素点后的图片
