class Shape:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def area(self):
        pass
    
    def name(self):
        print("I'm shape!")
        
class Circle(Shape):
    def __init__(self, x, y, r):
        super().__init__(x, y)
        self.r = r

    def area(self):
        return 3.14 * self.r ** 2

class Square(Shape):
    def __init__(self, x, y, l):
        super().__init__(x, y)
        self.l = l

    def area(self):
        return self.l ** 2

    def perimeter(self):
        return self.l * 4
    

def main():
    circle = Circle(0, 0, 5)
    print(circle.area())
    print(circle.name())

    square = Square(0, 0, 4)
    print(square.area())  
    print(square.perimeter())  
    print(square.name())

if __name__ == '__main__':
    main()

