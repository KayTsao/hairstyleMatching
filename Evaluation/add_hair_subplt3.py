import os
import sys
import shutil
import os
import sys
import glob
import shutil
from PIL import Image
def change_color(i_dir,o_dir):
    i = 1
    j = 1
    img = Image.open(i_dir)#读取照片
    width = img.size[0]#长度
    height = img.size[1]#宽度
    offset1=219
    offset2=438
    for i in range(319,502):#遍历所有长度的点
        for j in range(209,392):#遍历所有宽度的点
            data = (img.getpixel((i,j)))#打印该图片的所有点
            if (data[0] == 0 and data[1] == 0):#Blue Area
                img.putpixel((i,j),(0,0,0,125))#则这些像素点的颜色改成白色
            else:
                img.putpixel((i,j),(100,0,0,75))#则这些像素点的颜色改成透明红色
            img_data = (img.getpixel((i-offset1,j)))
            msk_data = (img.getpixel((i,j)))
            img.putpixel((i,j),(img_data[0]+msk_data[0], img_data[1]+msk_data[1], img_data[2]+msk_data[2], 255))

    for i in range(538,721):#遍历所有长度的点
        for j in range(209,392):#遍历所有宽度的点
            data = (img.getpixel((i,j)))#打印该图片的所有点
            if (data[0] == 0 and data[1] == 0):#Blue Area
                img.putpixel((i,j),(0,0,0,125))#则这些像素点的颜色改成白色
            else:
                img.putpixel((i,j),(100,0,0,75))#则这些像素点的颜色改成白色
            img_data = (img.getpixel((i-offset2,j)))
            msk_data = (img.getpixel((i,j)))
            img.putpixel((i,j),(img_data[0]+msk_data[0], img_data[1]+msk_data[1], img_data[2]+msk_data[2], 255))
    img.save(o_dir)
    return img
    

def parse_command_line():
    if len(sys.argv) < 2:
        display_help()
        exit(-1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    return input_dir, output_dir

if __name__ == '__main__':
    i_dir, o_dir = parse_command_line()

    path = i_dir
    files = os.listdir(path)
    total_num = len(files)
    s = []
    idx=0
    for f in files:
        idx +=1 
        src_path = i_dir + f
        dst_path = o_dir + f
        img = change_color(src_path, dst_path)#[0,255,0,255])   
        print('%i of %i' %(idx, total_num))



#img = img.convert("RGB")#把图片强制转成RGB
#img.save(o_dir)#保存修改像素点后的图片
