# 导入 Flask 核心组件
# render_template: 渲染 Jinja2 模板并返回 HTML 响应
# flash: 向用户显示一次性提示消息（存储在 session 中）
# redirect + url_for: 实现 PRG（Post-Redirect-Get）模式，防止表单重复提交
from flask import Flask, render_template, flash, redirect, url_for
# FlaskForm: Flask-WTF 提供的表单基类，自动集成 CSRF Token 生成与校验
from flask_wtf import FlaskForm
# WTForms 字段类型：StringField(文本输入)、PasswordField(密码输入)、SubmitField(提交按钮)
from wtforms import StringField, PasswordField, SubmitField
# DataRequired: 内置验证器，确保字段提交时不为空
from wtforms.validators import DataRequired
# Flask-Bootstrap: 为页面和表单提供 Bootstrap UI 框架支持
from flask_bootstrap import Bootstrap

# 创建 Flask 应用实例
app = Flask(__name__)
# 密钥：用于session、CSRF令牌加密，生产环境务必更换为随机字符串！
app.config['SECRET_KEY'] = 'Chapter4-change-in-production'

# 初始化 Flask-Bootstrap 扩展
bootstrap = Bootstrap(app)


class LoginForm(FlaskForm):
    """登录表单定义，继承 FlaskForm 自动获得隐藏的 CSRF Token 字段"""
    # 用户名输入框
    # label: 前端显示的标签文本
    # validators: 验证器列表，DataRequired 在提交时校验非空
    # message: 校验失败时显示的错误提示文案
    username = StringField(
        label='用户名',
        validators=[DataRequired(message='用户名不能为空')]
    )

    # 密码输入框（浏览器会自动遮蔽输入内容）
    password = PasswordField(
        label='密码',
        validators=[DataRequired(message='密码不能为空')]
    )

    # 提交按钮，label 即按钮上显示的文本
    submit = SubmitField(label='登录')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """处理登录页面的展示与表单提交"""
    # 每次请求创建新的表单实例
    # GET 请求：渲染空白表单；POST 请求：自动从 request.form 填充数据
    form = LoginForm()
    # validate_on_submit() = request.method == 'POST' and form.validate()
    if form.validate_on_submit():
        # 表单校验通过后执行的业务逻辑
        flash(f'登录成功！欢迎 {form.username.data}', 'success')
        # PRG 模式：重定向，防止刷新页面重复提交表单
        return redirect(url_for('login'))

    # GET 请求 或 POST校验失败时渲染模板
    return render_template('login.html', form=form)


if __name__ == '__main__':
    # debug=True 开启调试模式，仅开发使用
    app.run(debug=True)