class Student:
    # 类变量：所有实例共享
    school = "实验中学"
    student_count = 0

    def __init__(self, name, score):
        self.name = name      # 实例变量：每个学生独有
        self.score = 0        # 实例变量：初始化成绩
        Student.student_count += 1  # 创建实例时，总人数+1

        # 使用静态方法校验成绩后再赋值
        if self.validate_score(score):
            self.score = score
        else:
            print(f"{name} 的成绩 {score} 无效，默认设为0")

    def show_info(self):
        """打印学生信息"""
        print(f"[{self.school}] 姓名: {self.name}, 成绩: {self.score}分, 全校共{self.student_count}人")

    @classmethod
    def set_school(cls, new_name):
        """修改学校名称 (类方法)"""
        cls.school = new_name

    @staticmethod    
    def validate_score(score):
        """验证成绩是否在 0-100 之间 (静态方法)"""
        return 0 <= score <= 100

# --- 实例化测试 ---
print("--- 创建学生对象 ---")
s1 = Student("小明", 85)
s2 = Student("小红", 92)

print("\n--- 初始信息 ---")
s1.show_info()
s2.show_info()

print("\n--- 修改学校名称为 '实验高中' ---")
Student.set_school("实验高中")

print("\n--- 修改后信息 (观察类变量变化) ---")
s1.show_info()
s2.show_info()

print("\n--- 验证成绩校验 (105 和 85) ---")
# 创建一个成绩为105的学生
s3 = Student("小刚", 105)
s3.show_info() # 成绩应为0或默认值

print("\n--- 打印全校总人数 ---")
print(f"当前系统总人数: {Student.student_count}")
'''- @classmethod 装饰器表示这是类方法
- cls.school 修改的是 类变量 （所有实例共享）
- cls 参数指向类本身
class Student:
    def __init__(self, name):  # 构造函数（创建对象时调用）
        self.name = name
        print(f"{self.name} 创建了")
    
    def __del__(self):  # 析构函数（销毁对象时调用）
        print(f"{self.name} 销毁了")'''