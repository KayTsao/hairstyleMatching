import os
import sys
import shutil
import os
import sys
import glob
import shutil
from PIL import Image
def getAllFileInFolder(folderPath, fileExtension):
    totalExtension = ''
    if fileExtexsion.startwith('.'):
        totalExtension = fileExtension
    else
        totalExtension = '.' + fileExtension
def change_color(i_dir,color):
    i = 1
    j = 1
    img = Image.open(i_dir)#读取照片
    width = img.size[0]#长度
    height = img.size[1]#宽度
    for i in range(0,width):
        for j in range(0,height):
            data = (img.getpixel((i,j)))
            if (data[0] == 0 and data[1] == 0 and data[2] == 0):#white Area
                img.putpixel((i,j),(0,255,0,255))#则这些像素点的颜色改成Green
            else:
                img.putpixel((i,j),(0,0,0,0))#则这些像素点的颜色改成白色
    img.save(os.path.join('tmp/', i_dir))
    return img
    

def parse_command_line():
    if len(sys.argv) < 3:
        display_help()
        exit(-1)

    input1_dir = sys.argv[1]
    input2_dir = sys.argv[2]
    output_dir = sys.argv[3]
    return input1_dir, input2_dir, output_dir

if __name__ == '__main__':
    i_dir1, i_dir2, o_dir = parse_command_line()
    img1 = change_color(i_dir1, [0,255,0,255])
    img2 = change_color(i_dir2, [255,0,0,255])
    Image.blend(img1,img2,0.6).save(o_dir)



#img = img.convert("RGB")#把图片强制转成RGB
#img.save(o_dir)#保存修改像素点后的图片
