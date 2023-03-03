from shape import Circle, Square

import matplotlib.pyplot as plt

class CirclePlot(Circle):
    def __init__(self,x,y,r):
        super().__init__(x,y,r)
        self.center = (self.x,self.y)
        self.radius = self.r
    
    def plot(self, colour, fills):
        self.circle = plt.Circle(self.center, self.radius, color=colour, fill=fills)
        return self.circle

class SquarePlot(Square):
    def __init__(self, x, y, l):
        super().__init__(x, y, l)
        self.center = (self.x, self.y)
        self.height = self.l
        self.width = self.l
        
    
    def plot(self, colour, fills):
        self.square = plt.Rectangle(self.center, self.width, self.height, color=colour, fill=fills)
        return self.square
        
cp = CirclePlot(0,0,5)
cp = cp.plot("red", False)

s = SquarePlot(3,3,4)
sp = s.plot("blue", False)

a = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
a.add_patch(cp)        
a.add_patch(sp)        

a.set_aspect('equal') 

plt.show()