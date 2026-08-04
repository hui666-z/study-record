# 导入 Flask-WTF 表单基类，自动集成 CSRF Token 生成与校验
from flask_wtf import FlaskForm
# 导入 WTForms 字段类型：文本输入、单选按钮组、提交按钮
from wtforms import StringField, RadioField, SubmitField
# 导入验证器：DataRequired(非空校验)、Email(邮箱格式校验)
from wtforms.validators import DataRequired, Email
# 导入 Flask 核心组件：应用实例、模板渲染
from flask import Flask, render_template

class RegisterForm(FlaskForm):
    """用户注册表单类，继承 FlaskForm 自动获得隐藏的 CSRF Token 字段"""

    # 用户名输入框，DataRequired() 确保提交时不为空
    username = StringField(label='用户名', validators=[DataRequired(message="用户名不能为空")])

    # 邮箱输入框，配置双重验证器
    email = StringField(
        label='邮箱',
        validators=[
            DataRequired(message='邮箱不能为空'),       # 第一层：非空校验
            Email(message='请输入有效的邮箱地址')        # 第二层：邮箱格式校验
        ]
    )

    # 性别单选按钮组
    sex = RadioField(
        label='性别',
        validators=[DataRequired(message="请选择性别")],  # 必须选择一个选项
        choices=[  # 可选项列表，格式 (value, label)
            (1, '男'),
            (2, '女'),
        ],
        coerce=int  # 将前端提交的字符串强制转换为 int 类型
    )

    # 提交按钮
    submit = SubmitField(label='注册')


# 创建 Flask 应用实例
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # 生产环境更换为随机密钥


# 视图函数
@app.route('/', methods=['GET', 'POST'])
def register():
    """处理注册页面 GET展示 / POST表单提交"""
    form = RegisterForm()

    # 判断：POST请求 并且表单全部验证通过
    if form.validate_on_submit():
        # 获取表单数据
        print(f"用户名: {form.username.data}")
        print(f"邮箱: {form.email.data}")
        print(f"性别: {form.sex.data}")  # 输出整数 1 / 2
        return "注册成功！"

    # GET访问 或者表单校验失败，渲染页面
    return render_template('register.html', form=form)


if __name__ == '__main__':
    # debug=True仅开发使用
    app.run(debug=True)