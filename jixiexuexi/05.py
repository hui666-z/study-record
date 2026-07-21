# class Person:
#         def __init__(self, name, age):
#             self.name = name      # 公有属性
#             self._age = age       # 受保护属性（约定俗成，外部仍可访问但不建议）
#             self.__salary = 5000  # 私有属性（外部无法直接通过 self.__salary 访问）
    
#         def get_salary(self):
#             """提供公有方法来访问私有属性"""
#             return self.__salary
    
#         def set_salary(self, amount):
#             """提供公有方法来修改私有属性，可以加入逻辑判断"""
#             if amount > 0:
#                 self.__salary = amount
    
#     # 使用示例
# p = Person("Alice", 30)
# print(p.name)          # 输出: Alice
#     # print(p.__salary)    # 报错: AttributeError
# print(p.get_salary())  # 输出: 5000
    
#     # 添加空行以分隔不同的类定义
# class Animal:
#         def __init__(self, name):
#             self.name = name
    
#         def speak(self):
#             return f"{self.name} makes a sound."
    
#     # Dog 继承自 Animal
# class Dog(Animal):
#         def __init__(self, name, breed):
#             super().__init__(name)  # 调用父类的构造函数
#             self.breed = breed      # 子类特有的属性
    
#         def speak(self):
#             # 这里重写了父类的方法
#             return f"{self.name} barks!"
    
#     # 使用示例
# dog = Dog("Buddy", "Golden Retriever")
# print(dog.name)   # 输出: Buddy (继承自父类)
# print(dog.breed)  # 输出: Golden Retriever (子类特有)
# print(dog.speak())# 输出: Buddy barks! (调用子类重写后的方法)
    
#     # 添加空行以分隔不同的类定义
# class Calculator:
#         def add(self, a, b, c=0):
#             """
#             模拟重载：
#             如果传入2个参数，c默认为0，执行两数相加。
#             如果传入3个参数，执行三数相加。
#             """
#             return a + b + c
    
# calc = Calculator()
# print(calc.add(1, 2))    # 输出: 3 (相当于 add(1, 2, 0))
# print(calc.add(1, 2, 3)) # 输出: 6
    
#     # 添加空行以分隔不同的类定义
# class Greeter:
#         def greet(self, name):
#             if isinstance(name, str):
#                 return f"Hello, {name}!"
#             elif isinstance(name, list):
#                 return f"Hello, {' and '.join(name)}!"
#             else:
#                 return "Hello, Guest!"
    
# g = Greeter()
# print(g.greet("Alice"))       # 输出: Hello, Alice!
# print(g.greet(["Alice", "Bob"])) # 输出: Hello, Alice and Bob!
'''特性	说明	Python 实现方式
封装	隐藏内部细节，保护数据	使用 __private 变量，提供 getter/setter 方法
继承	代码复用，建立类层次结构	class Child(Parent):，使用 super() 调用父类
重载	同名方法不同实现	重写：子类直接定义同名方法；模拟重载：使用默认参数或类型检查'''
'''super() 参数传递的一致性,使用 *args, **kwargs 兼容未定义的初始化,MRO 顺序的影响,Mixin 类的规范'''
'''try：必选。包裹可能抛出异常的代码。
except：可选（但通常至少有一个）。捕获并处理特定类型的异常。
else：可选。当 try 块中没有发生任何异常时执行。
finally：可选。无论是否发生异常，最终都会执行的代码（常用于资源清理，如关闭文件、数据库连接等）。'''
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_config_file(filename):
    """
    演示 try-except-else-finally 的完整流程
    """
    f = None
    try:
        # 1. [TRY] 尝试打开文件（可能抛出 FileNotFoundError, PermissionError 等）
        print(f"--- 尝试打开文件: {filename} ---")
        f = open(filename, 'r', encoding='utf-8')
        content = f.read()
        
        # 模拟一个可能发生的其他错误（例如内容格式不对）
        if "error" in content:
            raise ValueError("文件内容包含非法标记 'error'")
            
    except FileNotFoundError:
        # 2. [EXCEPT] 捕获文件不存在的情况
        print(f"错误: 文件 '{filename}' 未找到。")
        return None # 即使 return，finally 也会执行
        
    except ValueError as e:
        # 2. [EXCEPT] 捕获值错误
        print(f"错误: 文件格式无效 - {e}")
        return None
        
    except Exception as e:
        # 2. [EXCEPT] 捕获其他所有未知异常
        logger.error(f"发生未知错误: {e}", exc_info=True)
        return None
        
    else:
        # 3. [ELSE] 只有当 try 完全成功（无异常）时才执行
        # 这里处理“成功读取后”的业务逻辑
        print("--- 文件读取成功，开始处理数据 ---")
        processed_data = content.upper() # 假设这是数据处理
        return processed_data
        
    finally:
        # 4. [FINALLY] 无论如何都会执行
        # 用于确保资源被释放
        print("--- 执行清理工作 (关闭文件) ---")
        if f and not f.closed:
            f.close()
            print("文件已关闭。")

# --- 测试运行 ---
if __name__ == "__main__":
    # 场景 A: 文件不存在
    print("\n=== 场景 A: 文件不存在 ===")
    result = process_config_file("non_existent.txt")
    print(f"结果: {result}")

    # 场景 B: 文件存在且正常
    # 先创建一个测试文件
    with open("test_config.txt", "w") as tf:
        tf.write("hello world")
    
    print("\n=== 场景 B: 正常文件 ===")
    result = process_config_file("test_config.txt")
    print(f"结果: {result}")

    # 场景 C: 文件存在但内容触发 ValueError
    with open("bad_config.txt", "w") as tf:
        tf.write("this has error in it")
        
    print("\n=== 场景 C: 内容错误 ===")
    result = process_config_file("bad_config.txt")
    print(f"结果: {result}")