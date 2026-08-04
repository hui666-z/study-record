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
        # request.args 是一个 ImmutableMultiDict，存储 URL 查询字符串中的参数
        # 例如: /?msg_get=hello → request.args.get('msg_get') 返回 'hello'
        # .get() 方法在键不存在时返回 None，避免抛出 KeyError
        msg_get = request.args.get('msg_get')

    # ========== 处理 POST 请求 ==========
    elif request.method == 'POST':
        # request.form 是一个 ImmutableMultiDict，存储 POST 请求体中的表单数据
        # 仅适用于 Content-Type 为 application/x-www-form-urlencoded 或 multipart/form-data 的请求
        # 若前端使用 JSON 提交，应改用 request.get_json()
        msg_post = request.form.get('msg_post')

    # 渲染 templates/form.html 模板
    # 将 Python 变量以关键字参数形式传入模板上下文
    # 模板中可通过 {{ msg_get }} 和 {{ msg_post }} 访问这些值
    return render_template(
        'form.html',
        msg_get=msg_get,
        msg_post=msg_post
    )


# 应用入口点
# 仅在直接运行此脚本时执行（被其他模块 import 时不执行）
if __name__ == '__main__':
    # debug=True 开启调试模式
    app.run(debug=True)