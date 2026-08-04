from flask import Flask, render_template, request

app = Flask(__name__)
# 设置密钥，防止session、flash相关报错
app.config['SECRET_KEY'] = 'your-secret-key'

# 自定义404错误页面处理器
@app.errorhandler(404)
def not_found_page(e):
    return render_template(
        'custom_error.html',
        title='Emmmm, 404!',
        description=f'搞错url了吧：{request.path}',
        status_code=404
    ), 404   # ✅重点：必须返回状态码，否则浏览器识别依旧是200

# 首页测试路由
@app.route('/')
def index():
    return '<h1>主页</h1><p><a href="/nonexistent">点击触发 404</a></p>'

if __name__ == '__main__':
    app.run(debug=True)