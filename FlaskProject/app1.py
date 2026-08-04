from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired
from flask_ckeditor import CKEditor, CKEditorField

# 创建Flask应用实例
app = Flask(__name__)
# 密钥，必须配置，用于CSRF保护、flash消息
app.config['SECRET_KEY'] = 'dev-secret-key-for-ckeditor'
# CKEditor配置：开启使用本地静态资源（不联网加载CDN编辑器）
app.config['CKEDITOR_SERVE_LOCAL'] = True
# 设置编辑器语言为简体中文
app.config['CKEDITOR_LANGUAGE'] = 'zh-cn'

# 初始化CKEditor扩展
ckeditor = CKEditor(app)

# ========== (1) 定义WTForms表单类 ==========
class ArticleForm(FlaskForm):
    # 标题输入框，DataRequired() 不能为空
    title = StringField('标题', validators=[DataRequired(message="标题不能为空")])
    # CKEditor富文本框字段
    content = CKEditorField('内容', validators=[DataRequired(message="内容不能为空")])
    # 提交按钮
    submit = SubmitField('查看')

# ========== (2) 文章编辑路由，支持GET、POST ==========
@app.route('/ckeditor_article', methods=['GET', 'POST'])
def ckeditor_article():
    # 实例化表单
    form = ArticleForm()
    # 判断：POST请求 并且表单校验全部通过
    if request.method == 'POST' and form.validate_on_submit():
        # 发送flash提示消息，分类success（成功样式）
        flash('文章提交成功！', 'success')
        # 渲染预览页面，把form对象传递给预览模板
        return render_template('ckeditor_view.html', form=form)

    # GET访问页面 / 表单校验失败：渲染编辑页面
    return render_template('ckeditor_edit.html', form=form)

# 首页路由
@app.route('/')
def index():
    return '''
    <h2 class="text-center mt-5">
        <a href="/ckeditor_article" class="btn btn-lg btn-primary">开始使用 CKEditor</a>
    </h2>
    '''

if __name__ == '__main__':
    # 启动调试服务器
    app.run(debug=True)