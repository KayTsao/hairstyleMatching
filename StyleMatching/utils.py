import math
class Point:
    def __init__(self,x=0,y=0):
        self.x=x
        self.y=y
    def getx(self):
        return self.x
    def gety(self):
        return self.y 

#定义直线函数   
class GenLine:
    def __init__(self,k=0,b=0):
        self.k = k
        self.b = b

    def frm2P(self, p1, p2):
        dx= p2.x-p1.x
        dy= p2.y-p1.y
        self.k = dy/dx
        self.b = p1.y - self.k*p1.x
        return self.k, self.b

    def frmKP(self, k, p1):
        self.k = k
        self.b = p1.y - k * p1.x
        return self.k, self.b
    


#设置点p1的坐标为（0,0）       
#p1=Point(0,0)
#设置点p2的坐标为（3,4）
#p2=Point(3,4)
#l=Getlen(p1,p2)
#获取两点之间直线的长度
#l.getlen()
