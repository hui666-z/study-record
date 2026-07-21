# 学生成绩管理系统（列表版本）
# 用列表存储学生信息，每个学生是一个字典
# students = [
#     {'name': 'Alice', 'chinese': 85, 'math': 90, 'english': 88},
#     {'name': 'Bob', 'chinese': 78, 'math': 82, 'english': 80},
#     {'name': 'Charlie', 'chinese': 92, 'math': 95, 'english': 90}
# ]

# def get_valid_score(prompt):
#     """
#     获取有效的成绩输入 (0-100)
#     :param prompt: 输入提示信息
#     :return: 合法的成绩浮点数
#     """
#     while True:
#         try:
#             score = float(input(prompt))
#             if 0 <= score <= 100:
#                 return score
#             else:
#                 print("成绩必须在 0 到 100 之间，请重新输入。")
#         except ValueError:
#             print("输入无效，请输入数字。")

# while True:
#     # 打印系统主菜单
#     print('\n' + '='*10 + ' 学生成绩管理系统 ' + '='*10)
#     print('1. 查看所有学生成绩')
#     print('2. 添加学生')
#     print('3. 修改学生成绩')
#     print('4. 删除学生')
#     print('5. 查找学生')
#     print('6. 统计平均分')
#     print('7. 退出系统')
    
#     choice = input('请选择操作 (1-7): ')
    
#     # 1. 查看所有学生
#     if choice == '1':
#         if not students:
#             print('暂无学生信息！')
#         else:
#             # 打印表头，使用格式化字符串对齐
#             print(f'\n{"姓名":<8}{"语文":<8}{"数学":<8}{"英语":<8}{"总分":<8}')#<左对齐，>右对齐，^居中对齐后面的数字为宽度包括字符数
#             print('-' * 50)
#             for student in students:
#                 total = student['chinese'] + student['math'] + student['english']
#                 print(f"{student['name']:<10}{student['chinese']:<10}{student['math']:<10}{student['english']:<10}{total:<10}")
    
#     # 2. 添加学生
#     elif choice == '2':
#         name = input('请输入学生姓名：').strip()
#         if not name:
#             print('姓名不能为空！')
#             continue
            
#         # 检查是否重名
#         found = False
#         for student in students:
#             if student['name'] == name:
#                 print('该学生已存在！')
#                 found = True
#                 break
        
#         if not found:
#             # 使用辅助函数获取合法成绩
#             chinese = get_valid_score('请输入语文成绩：')
#             math = get_valid_score('请输入数学成绩：')
#             english = get_valid_score('请输入英语成绩：')
            
#             # 构建新学生字典并添加到列表
#             new_student = {
#                 'name': name,
#                 'chinese': chinese,
#                 'math': math,
#                 'english': english
#             }
#             students.append(new_student)
#             print(f'学生 {name} 添加成功！')
    
#     # 3. 修改学生成绩
#     elif choice == '3':
#         name = input('请输入要修改的学生姓名：')
#         # 使用 next() 高效查找学生，若未找到则返回 None
#         target_student = next((s for s in students if s['name'] == name), None)
        
#         if target_student:
#             print(f'当前成绩 - 语文：{target_student["chinese"]}, 数学：{target_student["math"]}, 英语：{target_student["english"]}')
#             chinese = get_valid_score('请输入新的语文成绩：')
#             math = get_valid_score('请输入新的数学成绩：')
#             english = get_valid_score('请输入新的英语成绩：')
            
#             # 更新字典中的成绩
#             target_student['chinese'] = chinese
#             target_student['math'] = math
#             target_student['english'] = english
#             print(f'学生 {name} 的成绩已更新！')
#         else:
#             print('未找到该学生！')
    
#     # 4. 删除学生
#     elif choice == '4':
#         name = input('请输入要删除的学生姓名：')
#         target_student = next((s for s in students if s['name'] == name), None)
        
#         if target_student:
#             students.remove(target_student)
#             print(f'学生 {name} 已删除！')
#         else:
#             print('未找到该学生！')
    
#     # 5. 查找学生
#     elif choice == '5':
#         name = input('请输入要查找的学生姓名：')
#         target_student = next((s for s in students if s['name'] == name), None)
        
#         if target_student:
#             total = target_student['chinese'] + target_student['math'] + target_student['english']
#             avg = total / 3
#             print(f'\n')
#             print(f'姓名：{target_student["name"]:<10} 语文：{target_student["chinese"]:<10} 数学：{target_student["math"]:<10} 英语：{target_student["english"]:<10} 总分：{total:<10} 平均分：{avg:.2f}')
#         else:
#             print('未找到该学生！')
    
#     # 6. 统计平均分
#     elif choice == '6':
#         if not students:
#             print('暂无学生信息！')
#         else:
#             # 使用生成器表达式计算各科平均分
#             chinese_avg = sum(s['chinese'] for s in students) / len(students)
#             math_avg = sum(s['math'] for s in students) / len(students)
#             english_avg = sum(s['english'] for s in students) / len(students)
            
#             print(f'\n')    
#             print(f'语文平均分：{chinese_avg:<10.2f} 数学平均分：{math_avg:<10.2f} 英语平均分：{english_avg:<10.2f}')#若有了：且有两种：型的<放前面
            
          
    
#     # 7. 退出
#     elif choice == '7':
#         print('感谢使用，再见！')
#         break
    
#     else:
#         print('无效选项，请重新选择！')     


print('-'*10+'用户界面'+'-'*10)

users = {
    'hui': '123'# 用户名: 密码(有空格)
}

logged_in = False      # 标记是否已登录
current_user = None    # 当前登录的用户名

while True:
    if not logged_in:
        print('\n----欢迎使用系统----')
        print('1. 登录')
        print('2. 注册')
        print('3. 退出')
        choice = input('请选择操作(1/2/3): ')
        
        if choice == '1':
            username = input('请输入用户名：')
            password = input('请输入密码：')
            if username in users and users[username] == password:
                print('登录成功！')
                logged_in = True
                current_user = username
            else:
                print('用户名或密码错误，请重试。')
                
        elif choice == '2':
            username = input('请输入新用户名：')
            if username in users:
                print('该用户名已存在，请选择其他用户名。')
            else:
                password = input('请输入密码：')
                users[username] = password
                print('注册成功！')
                
        elif choice == '3':
            print('感谢使用，再见！')
            break
            
        else:
            print('无效选项，请重新输入。')
            
    else:
        # 已登录后的菜单
        print(f'\n--- 用户 [{current_user}] 已登录 ---')
        print('4. 计算器')
        print('5. 权限检查')
        print('6. 退出登录')
        print('7. 退出系统')
        print('8. 注销账号')
        print('9. 查找账号')
        choice = input('请选择操作(4/5/6/7): ')
        
        if choice == '4':
            print("简易计算器（仅支持两个数的基本运算）")
            try:
                a = float(input("请输入第一个数: "))
                op = input("请输入运算符 (+, -, *, /): ")
                b = float(input("请输入第二个数: "))
                if op == '+':
                    print("结果:", a + b)
                elif op == '-':
                    print("结果:", a - b)
                elif op == '*':
                    print("结果:", a * b)
                elif op == '/':
                    if b != 0:
                        print("结果:", a / b)
                    else:
                        print("错误：除数不能为零！")
                else:
                    print("不支持的运算符")
            except ValueError:   
                print("输入无效，请输入数字。")
                
        elif choice == '5':
            # 示例：只有 hui 是管理员
            if current_user == 'hui':
                print("【权限】您是管理员，拥有全部权限。")
            else:
                print("【权限】您是普通用户。")
                
        elif choice == '6':
            print(f"用户 {current_user} 已退出登录。")
            logged_in = False
            current_user = None
            
        elif choice == '7':
            print('感谢使用，再见！')
            break
        elif choice == '8':
            username = input('请输入用户名：')
            if username in users:
                del users[username]
                print(f'用户 {username} 已注销。')
            else:
                print('用户不存在。')
        elif choice == '9':
            username = input('请输入用户名：')
            if username in users:
                print(f'用户 {username} 已存在。')
            else:
                print(f'用户 {username} 不存在。')
                
        else:
            print('无效选项，请重新输入。')        
