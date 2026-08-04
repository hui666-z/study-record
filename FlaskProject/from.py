# 导入 Flask 核心类及常用工具函数
from flask import Flask, render_template, request

# 创建 Flask 应用实例
app = Flask(__name__)

# 定义路由 '/'，同时允许 GET 和 POST 两种 HTTP 方法
# GET: 用于首次访问页面或提交搜索/筛选表单（参数在 URL 中）
# POST: 用于提交敏感数据或修改服务器状态的操作（参数在请求体中）
@app.route('/', methods=['GET', 'POST'])
def form():
    # 初始化两个变量，用于存储从请求中提取的数据
    # 默认为 None，当对应请求中没有该字段时，模板可据此判断是否显示结果
    msg_get = None
    msg_post = None

    # ========== 处理 GET 请求 ==========
    if request.method == 'GET':
        # request.args 获取url上的参数
        msg_get = request.args.get('msg_get')

    # ========== 处理 POST 请求 ==========
    elif request.method == 'POST':
        # request.form 获取表单post提交的数据
        msg_post = request.form.get('msg_post')

    # 渲染模板，传递变量到前端
    return render_template(
        'form.html',
        msg_get=msg_get,
        msg_post=msg_post
    )

# 应用入口点
if __name__ == '__main__':
    # debug=True 开启调试模式，代码修改自动重启服务
    app.run(debug=True)