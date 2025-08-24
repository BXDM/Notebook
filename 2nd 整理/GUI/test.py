import sys
import sqlite3
from PyQt5 import QtWidgets, QtCore
from datetime import datetime

class ReplacementApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_db()

    def init_ui(self):
        # 设置窗口标题和大小
        self.setWindowTitle("替代料输入")
        self.setGeometry(100, 100, 400, 400)

        # 创建输入框和标签
        self.hhpn_label = QtWidgets.QLabel("HHPN:")
        self.hhpn_input = QtWidgets.QLineEdit(self)

        self.aml_label = QtWidgets.QLabel("AML:")
        self.aml_input = QtWidgets.QLineEdit(self)

        self.amlpn_label = QtWidgets.QLabel("AMLPN:")
        self.amlpn_input = QtWidgets.QLineEdit(self)

        self.isin_bom_label = QtWidgets.QLabel("isin_bom:")
        self.isin_bom_input = QtWidgets.QLineEdit(self)

        self.record_label = QtWidgets.QLabel("Record:")
        self.record_input = QtWidgets.QLineEdit(self)

        self.spec_label = QtWidgets.QLabel("SPEC:")
        self.spec_input = QtWidgets.QLineEdit(self)

        # 按钮
        self.add_button = QtWidgets.QPushButton("插入替代料", self)
        self.add_button.clicked.connect(self.add_replacement)

        self.no_replacement_button = QtWidgets.QPushButton("无替代料推荐", self)
        self.no_replacement_button.clicked.connect(self.add_no_replacement)

        # 布局
        layout = QtWidgets.QFormLayout()
        layout.addRow(self.hhpn_label, self.hhpn_input)
        layout.addRow(self.aml_label, self.aml_input)
        layout.addRow(self.amlpn_label, self.amlpn_input)
        layout.addRow(self.isin_bom_label, self.isin_bom_input)
        layout.addRow(self.record_label, self.record_input)
        layout.addRow(self.spec_label, self.spec_input)
        layout.addWidget(self.add_button)
        layout.addWidget(self.no_replacement_button)

        self.setLayout(layout)

    def init_db(self):
        # 初始化数据库连接
        self.conn = sqlite3.connect("test.db")
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS Relation (
            HHPN TEXT, 
            AML TEXT, 
            AMLPN TEXT, 
            isin_bom INTEGER, 
            last_updated TEXT, 
            record TEXT, 
            SPEC TEXT,
            PRIMARY KEY (HHPN, AML, AMLPN)
        )''')
        self.conn.commit()

    def add_replacement(self):
        # 获取输入的值，并替换为空的字段
        hhpn = self.hhpn_input.text() or ""
        aml = self.aml_input.text() or ""
        amlpn = self.amlpn_input.text() or ""
        isin_bom = self.isin_bom_input.text() or "0"  # 可以根据需要设置默认值
        record = self.record_input.text() or ""
        spec = self.spec_input.text() or ""
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute("INSERT INTO Relation (HHPN, AML, AMLPN, isin_bom, last_updated, record, SPEC) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (hhpn, aml, amlpn, isin_bom, last_updated, record, spec))
            self.conn.commit()
            self.show_message("插入成功", "green")
        except sqlite3.IntegrityError:
            self.show_message("数据已存在，插入失败", "red")

    def add_no_replacement(self):
        hhpn = self.hhpn_input.text() or ""
        aml = self.aml_input.text() or ""
        amlpn = "DDDD"
        isin_bom = 9
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute("INSERT INTO Relation (HHPN, AML, AMLPN, isin_bom, last_updated) VALUES (?, ?, ?, ?, ?)",
                                (hhpn, aml, amlpn, isin_bom, last_updated))
            self.conn.commit()
            self.show_message("无替代料推荐已成功插入", "blue")
        except sqlite3.IntegrityError:
            self.show_message("数据已存在，插入失败", "red")


        def closeEvent(self, event):
            # 关闭数据库连接
            self.conn.close()

# 运行应用程序
app = QtWidgets.QApplication(sys.argv)
window = ReplacementApp()
window.show()
sys.exit(app.exec_())
