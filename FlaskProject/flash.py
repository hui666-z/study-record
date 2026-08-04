from flask import Flask, render_template, flash

app = Flask(__name__)
# 设置 SECRET_KEY，flash依赖session，必须配置
app.config['SECRET_KEY'] = 'dev-secret-key-change-in-production'


@app.route('/')
def index():
    """根路由，首页入口"""
    return '<h1>Flask Flash 测试</h1><p><a href="/flash">点击查看 Flash 消息</a></p>'


@app.route('/flash')
def flash_message():
    """Flash 消息测试路由"""
    # flash(消息内容, 分类标签)
    flash('这是一条成功消息', 'success')
    flash('这是一条危险消息', 'danger')
    flash('这是一条信息消息', 'info')
    flash('这是一条警告消息', 'warning')
    return render_template('flash.html')


if __name__ == '__main__':
    app.run(debug=True)