import sys
import sqlite3
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QDateTimeEdit, QGroupBox, QMessageBox
from PyQt5.QtCore import QDateTime
from datetime import datetime

# 连接SQLite数据库
conn = sqlite3.connect('test.db')
cursor = conn.cursor()

# 获取当前时间，格式化为字符串
def get_current_time():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# 插入数据函数
def insert_data(data):
    try:
        cursor.execute("BEGIN")
        # 插入每个表的数据
        if data['Supplier']:
            cursor.execute(
                "INSERT INTO Supplier (AML, vendor_code, payment_term, delivery_term, last_updated) VALUES (?, ?, ?, ?, ?)",
                (*data['Supplier'], get_current_time())
            )
        if data['MaterialPartNumber']:
            cursor.execute(
                "INSERT INTO MaterialPartNumber (HHPN, description, customer) VALUES (?, ?, ?)",
                (*data['MaterialPartNumber'],)
            )
        if data['Project']:
            cursor.execute(
                "INSERT INTO Project (project, application, status, forecast, mp_schedule) VALUES (?, ?, ?, ?, ?)",
                (*data['Project'],)
            )
        # 其他表的插入逻辑可按照相似方式添加
        cursor.execute("COMMIT")
        QMessageBox.information(None, "成功", "数据已成功插入！")
    except Exception as e:
        cursor.execute("ROLLBACK")
        QMessageBox.critical(None, "错误", f"插入数据时发生错误：{str(e)}")

# 创建每个表的输入区域
def create_group_box(table_name, fields):
    group_box = QGroupBox(table_name)
    layout = QFormLayout()
    inputs = {}

    for field in fields:
        if field not in ['last_updated', 'update']:
            label = QLabel(field)
            line_edit = QLineEdit()
            inputs[field] = line_edit
            layout.addRow(label, line_edit)

    group_box.setLayout(layout)
    return group_box, inputs

# 主窗口类
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('数据库数据插入界面')
        self.setGeometry(200, 200, 800, 600)
        self.layout = QVBoxLayout()

        # 存储每个表的输入字段
        self.data = {
            'Supplier': [],
            'MaterialPartNumber': [],
            'Project': [],
            # 其他表可以继续添加
        }
        self.inputs = {}

        # 根据表结构创建输入区域
        self.create_ui()

        # 提交按钮
        submit_button = QPushButton('提交数据')
        submit_button.clicked.connect(self.submit_data)
        self.layout.addWidget(submit_button)

        self.setLayout(self.layout)

    def create_ui(self):
        # 为每个表创建输入区域
        supplier_group, supplier_inputs = create_group_box('Supplier', ['AML', 'vendor_code', 'payment_term', 'delivery_term'])
        self.layout.addWidget(supplier_group)
        self.inputs['Supplier'] = supplier_inputs

        material_part_group, material_part_inputs = create_group_box('MaterialPartNumber', ['HHPN', 'description', 'customer'])
        self.layout.addWidget(material_part_group)
        self.inputs['MaterialPartNumber'] = material_part_inputs

        project_group, project_inputs = create_group_box('Project', ['project', 'application', 'status', 'forecast', 'mp_schedule'])
        self.layout.addWidget(project_group)
        self.inputs['Project'] = project_inputs

        # 其他表的输入区域可以在这里继续添加

    def submit_data(self):
        # 收集输入的数据
        data = {}
        for table, inputs in self.inputs.items():
            data[table] = [input_field.text() for input_field in inputs.values()]

        # 插入数据到数据库
        insert_data(data)

# 主程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

# 关闭数据库连接
conn.close()

#%%
