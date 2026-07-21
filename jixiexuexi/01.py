'''
a=1
b=1.1
d=True
e='123'
f=[1,2,3]
h={'key'}
g=(1,)
j={'name': 'hui'}
print(type(a))
print(type(b))
'''
'''
print(type(d))
print(type(e))
print(type(f))
print(type(h))
print(type(g))
print(type(j))
print(f"我是{d},明天{a}岁了")
print("我是%s,明天%d岁了"%(d,a))
print(f"我是{d},体重到了{b}公斤")
print(f"我是{d},体重到了%f公斤"%b)
print(f"我是{d},体重到了{b:.3f}公斤")
print(f"我是{d},体重到了{b:.1f}

print('-'*10+'用户界面'+'-'*10)
'''

'''
d='abcdefghijklmnopqrstuvwxyz'#索引str型从0开始 d[start:stop:step]切片不能都为负数
print(d[0])  # a
print(d[1])  # b
print(d[2:5])  # cde
print(d[:5])  # abcde
print(d[0:5])  # abcde
print(d[0:5:1])#ace
print(d[5:])  # fghijklmnopqrstuvwxyz
print(d[-1])  # z负数从后往前索引
print(d[-2])  # y
print(d[:])#从头到尾
print(d[::-1])#从尾到头
print(d[::-2])#从尾到头，步长为2
print(d[::2])#从0开始，步长为2
print(d.find('c'))#返回c在d中的索引位置,找不到返回-1
print(d.index('c'))#返回c在d中的索引位置，找不到会报错
print(d.count('c'))#统计c在d中出现的次数
print(d.upper())#将d中的小写字母转换为大写字母忽略大小写
print(d.lower())#将d中的大写字母转换为小写字母
print(d.replace('a','b'))#将d中的a替换为b,返回一个新的字符串，原字符串不变
print(d.split('c'))#将d按照c分割成一个列表，返回一个新的列表，原字符串不变，但分隔符会消失
print(d.split('c',1))#将d按照c分割成一个列表，分割次数为1，返回一个新的列表，原字符串不变
print(d.strip('a'))#将d中的a去掉，返回一个新的字符串，原字符串不变
print(d.strip('a','b'))#将d中的a和b去掉，返回新的字符串，原字符串不变
print(d.lstrip('a'))#将d中的a去掉，返回一个新的字符串，原字符串不变
print(d.rstrip('a'))#将d中的a去掉，返回一个新的字符串，原字符串不变
print(d.startswith('a'))#判断d是否以a开头，返回布尔值，原字符串不变
print(d.endswith('a'))#判断d是否以a结尾，返回布尔值，原字符串不变
print(d.isalpha())#判断d是否只包含字母，返回布尔值，原字符串不变
print(d.isdigit())#判断d是否只包含数字，返回布尔值，原字符串不变
print(d.isalnum())#判断d是否只包含字母和数字，返回布尔值，原字符串不变
print(d.isupper())#判断d是否只包含大写字母，返回布尔值，原字符串不变
'''


# ... existing code ...
# date = '8,3,5,3,9,1,5,7,1,9'  # 定义一个包含数字的字符串，数字之间用逗号分隔
# date01 = []  # 创建一个空列表，用于存储转换后的整数
# for i in date.split(','):  # 使用逗号分割字符串，遍历每个数字字符串
#     date01.append(int(i))  # 将每个数字字符串转换为整数，并添加到列表中
# setdate = set(date01)  # 将列表转换为集合，自动去除重复的数字
# print(date01)  # 打印转换后的整数列表
# print(min(setdate))  # 打印集合中的最小值
# print(max(setdate))  # 打印集合中的最大值
# #for index, i in enumerate(setdate):  # 遍历集合并获取每个元素的索引和值
# #    print(f'索引: {index}, 值: {i}')  # 打印每个元素的索引和对应的值

# # 将排序后的集合转换为字典：键是索引（从0开始），值是排序后的数字
# s = {index: value for index, value in enumerate(sorted(setdate))}
# print(s)  # 打印生成的字典，例如: {0: 1, 1: 3, 2: 5, 3: 7, 4: 8, 5: 9}
# [x for x in setdate]  # 列表推导式，生成一个包含集合中所有元素的列表
# def calculate_cart(items, discount=1.0, *args, **kwargs):
#     """
#     计算购物车总价
    
#     参数:
#     items: 商品列表，每个元素是元组 (商品名, 单价, 数量)
#     discount: 折扣率，默认为1.0（无折扣）
#     *args: 额外的优惠券金额
#     **kwargs: 其他费用（如运费、包装费等）
    
#     返回:
#     最终应付金额
#     """
#     # 计算商品小计
#     subtotal = 0
#     for item in items:
#         name, price, quantity = item
#         item_total = price * quantity
#         subtotal += item_total
#         print(f"商品: {name}, 单价: {price}, 数量: {quantity}, 小计: {item_total:.2f}")
    
#     print(f"\n商品总计: {subtotal:.2f}")
    
#     # 应用折扣率
#     after_discount = subtotal * discount
#     if discount != 1.0:
#         print(f"折扣率: {discount}, 折后金额: {after_discount:.2f}")
    
#     # 减去优惠券金额
#     coupon_total = sum(args)
#     if coupon_total > 0:
#         after_discount -= coupon_total
#         print(f"优惠券减免: {coupon_total:.2f}")
    
#     # 确保金额不为负数
#     if after_discount < 0:
#         after_discount = 0
    
#     # 加上其他费用
#     extra_fees = sum(kwargs.values())
#     if extra_fees > 0:
#         after_discount += extra_fees
#         print("其他费用:")
#         for fee_name, fee_amount in kwargs.items():
#             print(f"  {fee_name}: {fee_amount:.2f}")
    
#     final_amount = after_discount
#     print(f"\n最终应付金额: {final_amount:.2f}")
    
#     return final_amount


# # 测试示例
# if __name__ == "__main__":
#     定义商品列表
#     cart_items = [
#         ("苹果", 5.5, 3),
#         ("香蕉", 3.2, 5),
#         ("牛奶", 12.8, 2),
#         ("面包", 8.5, 1)
#     ]
    
#     print("=" * 40)
#     print("示例1: 基础购物（无折扣，无优惠券）")
#     print("=" * 40)
#     total1 = calculate_cart(cart_items)
    
#     print("\n" + "=" * 40)
#     print("示例2: 使用折扣和优惠券")
#     print("=" * 40)
#     total2 = calculate_cart(cart_items, 0.9, 10, 5)
    
#     print("\n" + "=" * 40)
#     print("示例3: 完整功能（折扣+优惠券+额外费用）")
#     print("=" * 40)
#     total3 = calculate_cart(
#         cart_items,
#         0.85,
#         20, 15,  
#         shipping=10,
#         packing=5,
#         insurance=3
#     )
    
#     print("\n" + "=" * 40)
#     print("示例4: 仅位置参数和默认参数")
#     print("=" * 40)
#     simple_items = [("书", 45.0, 2)]
#     total4 = calculate_cart(simple_items, discount=0.95)

# 定义一个名为 Student 的类
# class 关键字用于定义类，类名通常采用大驼峰命名法（每个单词首字母大写）
# class Student:
#     """
#     学生类：用于存储学生信息并执行相关操作
#     """

#     # 类变量：所有实例共享的数据（例如统计学生总数）
#     student_count = 0

#     # __init__ 是构造函数（初始化方法）
#     # 当创建一个新的 Student 对象时，这个方法会自动被调用
#     # self 代表当前实例对象本身，必须作为第一个参数
#     def __init__(self, name, age, grade):
#         """
#         初始化学生对象
#         :param name: 姓名 (字符串)
#         :param age: 年龄 (整数)
#         :param grade: 年级 (字符串)
#         """
#         # 实例变量：每个对象独有的数据
#         # 使用 self.name 将传入的 name 绑定到当前对象上
#         self.name = name
#         self.age = age
#         self.grade = grade
        
#         # 每创建一个学生，计数器加 1
#         Student.student_count += 1

#     # 实例方法：描述对象能做什么
#     def introduce(self):
#         """
#         打印学生的自我介绍
#         """
#         # 使用 f-string 格式化输出，访问当前对象的属性
#         print(f"大家好，我叫{self.name}，今年{self.age}岁，正在读{self.grade}年级。")

#     def have_birthday(self):
#         """
#         模拟过生日：年龄加 1
#         """
#         self.age += 1
#         print(f"{self.name} 过生日啦！现在 {self.age} 岁了。")

#     # 静态方法：不需要访问实例属性或类属性的方法
#     # 使用 @staticmethod 装饰器
#     @staticmethod
#     def is_adult(age):
#         """
#         判断是否成年
#         :param age: 年龄
#         :return: 布尔值
#         """
#         return age >= 18
#     '''
#     不依赖实例状态：该方法没有 self 参数，因此无法访问或修改特定实例的属性（如 self.name 或 self.age）。
# 不依赖类状态：该方法也没有 cls 参数，因此无法访问或修改类级别的属性或调用其他类方法。
# 纯逻辑函数：它仅仅接收一个外部传入的参数 age，并基于这个参数进行独立的逻辑判断（age >= 18），返回布尔值。
# 逻辑归属：虽然 is_adult 可以作为一个普通的独立函数存在，但因为它在逻辑上与“人”或“用户”的概念紧密相关，将其放在类内部可以使代码组织更清晰，表明这是该类相关的一个工具性功能。
# 性能与清晰性：明确告诉阅读代码的人（以及解释器），这个方法不需要实例化对象即可调用，且不会副作用地改变对象状态。
# 问ai类，对象，类变量，类函数的区别来了解
# 类 (Class)
# 定义：类是一个模板或蓝图。它定义了某一类事物共有的属性（数据）和行为（方法）。
# 特点：类本身不占用内存存储具体数据，它只是描述“是什么”
# 对象 (Object / Instance)
# 定义：对象是类的实例化。它是根据类这个模板创建出来的具体实体。
# 特点：每个对象都有自己独立的内存空间，存储具体的属性值。
# 类变量 (Class Variable)
# 定义：定义在类内部、但在任何方法之外的变量。
# 特点：
# 共享性：被该类的所有对象共享。如果一个对象修改了类变量（且该变量是不可变类型如整数、字符串），通常会影响其他对象看到的值（除非对象自身创建了同名实例变量遮蔽它）。
# 访问方式：可以通过 类名.变量名 或 对象.变量名 访问。
# 函数：
# 实例方法           self	默认类型。可以访问实例变量 (self.xxx) 和类变量。
# 类方法@staticmethod cls	接收类本身作为第一个参数。可以访问类变量，但不能直接访问实例变量。
# 静态方法@staticmethod 无	不接收 self 或 cls。无法访问实例变量或类变量（除非显式通过类名调用）
# Student.student_count类变量 s1.student_count实例化对象的
#     '''
#     # 魔术方法（特殊方法）：__str__ 用于定义打印对象时的显示内容
#     def __str__(self):
#         """
#         返回对象的字符串表示，方便调试和打印
#         """
#         return f"Student(name='{self.name}', age={self.age}, grade='{self.grade}')"


# # --- 主程序部分 ---

# if __name__ == "__main__":
#     print("--- 面向对象编程示例 ---")

#     # 1. 实例化对象（创建学生）
#     # 语法：对象名 = 类名(参数...)
#     # 这里调用了 __init__ 方法
#     student1 = Student("小明", 15, "初三")
#     student2 = Student("小红", 17, "高二")

#     # 2. 调用实例方法
#     # 语法：对象名.方法名()
#     student1.introduce()
#     student2.introduce()

#     print("-" * 20)

#     # 3. 修改对象状态
#     student1.have_birthday()

#     print("-" * 20)

#     # 4. 访问实例变量
#     # 语法：对象名.变量名
#     print(f"{student1.name} 的新年龄是: {student1.age}")

#     print("-" * 20)

#     # 5. 调用静态方法
#     # 静态方法可以通过类名直接调用，也可以通过实例调用
#     print(f"小明是否成年? {Student.is_adult(student1.age)}")
#     print(f"小红是否成年? {Student.is_adult(student2.age)}")

#     print("-" * 20)

#     # 6. 访问类变量
#     print(f"目前共有学生: {Student.student_count} 人")

#     print("-" * 20)

#     # 7. 打印对象（触发 __str__ 方法）
#     print(student1)class Student:
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age  # 这里会触发下面的 setter

    @property
    def age(self):
        """获取年龄"""
        return self._age  # 实际存储在 _age 中

    @age.setter
    def age(self, value):
        """设置年龄，包含逻辑验证"""
        if value < 0 or value > 150:
            raise ValueError("年龄不合法")
        self._age = value

# 使用示例
if __name__ == "__main__":
    s = Student("小红", 17)
    print(s.age)      # 触发 getter，输出 17
    s.age = 18        # 触发 setter，合法
    # s.age = -5      # 触发 setter，抛出 ValueError
    print(f"修改后的年龄: {s.age}")
# 1. 单下划线 _variable (保护成员 / Protected)
# 定义：变量或方法名前加一个下划线，如 _name。
# 含义：这是一种约定俗成的规范。它告诉其他开发者：“这是一个内部使用的变量，请在类外部不要直接访问或修改它。”
# 实际效果：
# 没有任何技术限制。你仍然可以在类外部通过 obj._name 访问和修改它。
# IDE（如 PyCharm, VS Code）通常会给出警告，提示这是受保护的成员。
# 适用场景：用于子类继承时，表示该成员仅供类内部或子类使用。
# 2. 双下划线 __variable (私有成员 / Private)
# 定义：变量或方法名前加两个下划线，如 __age。
# 含义：意图是“完全私有”，仅限类内部访问。
# 实际效果（名称修饰 Name Mangling）：
# Python 解释器会自动将变量名重命名为 _ClassName__variable。
# 例如，在 Student 类中定义 self.__secret = "password"，外部无法通过 student1.__secret 访问，会报错 AttributeError。
# 但你仍然可以通过 student1._Student__secret 强行访问（虽然强烈不建议这样做）。
# 目的：主要是为了避免子类中的命名冲突，而不是为了真正的安全隐藏。
# 适用场景：当你确定某个属性绝对不应该被子类覆盖或在外部被意外修改时使用。
# 3. 前后双下划线 __variable__ (魔术方法 / Special Methods)
# 定义：如 init, str, _conda create -n myenv python=3.9_len__。
# 含义：这些是 Python 的特殊方法（杜nder methods），由 Python 内部调用。
# 注意：不要自己发明这种命名的变量作为私有变量，这会与 Python 内部机制冲突。