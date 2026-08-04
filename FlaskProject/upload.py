# 导入 Flask 核心组件
# send_from_directory: 安全地从指定目录发送文件（自动防止路径遍历）
# request: 全局请求对象，用于获取上传文件和表单数据
# render_template: 渲染 Jinja2 模板并返回 HTML 响应
from flask import Flask, send_from_directory, request, render_template
from flask_bootstrap import Bootstrap
import os

app = Flask(__name__)
bootstrap = Bootstrap(app)

# ========== 上传目录配置 ==========
# 获取当前脚本所在目录的绝对路径，作为项目根目录基准
basedir = os.path.abspath(os.path.dirname(__file__))
# 拼接出上传文件的存储路径：项目根目录/uploads/
app.config['UPLOAD_FOLDER'] = os.path.join(basedir, 'uploads')
# 自动创建上传目录，exist_ok=True 表示目录已存在时不抛出异常
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# ========== 文件访问路由 ==========
@app.route('/uploads/<filename>')
def uploaded_files(filename):
    """根据文件名从上传目录中读取并返回文件内容"""
    path = app.config['UPLOAD_FOLDER']
    # send_from_directory 内部已做路径安全检查，防止 ../ 等路径遍历攻击
    return send_from_directory(path, filename)


# ========== 文件上传路由 ==========
@app.route('/', methods=['GET', 'POST'])
def upload():
    upload_dir = app.config['UPLOAD_FOLDER']
    # 读取已有文件列表
    files = os.listdir(upload_dir)

    if request.method == 'POST':
        # 从 POST 请求体中获取 name="upload" 的文件对象
        f = request.files.get('upload')
        if f:  # 判断是否选择了文件
            save_path = os.path.join(upload_dir, f.filename)
            f.save(save_path)
        # 重新刷新文件列表
        files = os.listdir(upload_dir)

    # GET 请求 / POST上传完成都渲染页面
    return render_template('upload.html', files=files)


if __name__ == '__main__':
    # debug=True 仅用于开发环境，生产环境必须关闭
    app.run(debug=True)