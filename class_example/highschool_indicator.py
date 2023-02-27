class mapo_HS:
    def __init__(self, grade, class_num):
        self.grade = grade
        self.class_num = class_num
        
    def class_indecator(self):
        return f"저는 {self.grade}학년 {self.class_num}반 학생입니다."
    
    def name(self, name):
        return f"저는 서울고등학교 학생 {name}입니다."
    
    
def main():
    while True:
        try:
            a = int(input("학년을 입력하세요>>>"))
            b = int(input("반을 입력하세요>>>"))
            c = input("이름을 입력하세요>>>")
            mapo = mapo_HS(a,b)
    
            ans = input("학년&반, 이름 중 하나를 고르시오>>>")
            
            if ans == "학년&반":
                print(mapo.class_indecator())
                break
                
            elif ans == '이름':
                print(mapo.name(c))
                break
            
            else:
                print("다시 입력해주세요!")
        except:
            print("다시 입력해주세요!")
            
            
main()


