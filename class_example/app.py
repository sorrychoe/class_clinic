from shape import Circle, Square

import matplotlib.pyplot as plt
    

class CirclePlot(Circle):
    def __init__(self,x,y,r):
        super().__init__(x,y,r)
    
    def plot(self, colour:str, fill:bool):
        self.circle = plt.Circle((self.x, self.y),self.r, colour, fill)
        return self.circle

class SquarePlot(Square):
    def __init__(self, x, y, l):
        super().__init__(x, y, l)
    
    def plot(self, colour:str, fill:bool):
        self.square = plt.Rectangle((self.x, self.y), self.l, colour, fill)
        return self.square
        
cp = CirclePlot(0,0,5)
cp = cp.plot("red", False)

s = SquarePlot(3,3,4)
sp = s.plot("red", False)

a = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
a.add_patch(sp)        
a.set_aspect('equal') 

plt.show()