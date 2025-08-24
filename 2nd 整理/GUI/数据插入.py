# Saving the provided code to a .py file as requested

import sys
import sqlite3
import json
from PyQt5 import QtWidgets, QtCore, QtGui
from datetime import datetime

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("界面设置")
        self.setGeometry(150, 150, 300, 200)

        # 创建字体选择和字体大小输入
        self.font_label = QtWidgets.QLabel("字体:")
        self.font_combo = QtWidgets.QFontComboBox(self)

        self.font_size_label = QtWidgets.QLabel("字体大小:")
        self.font_size_spin = QtWidgets.QSpinBox(self)
        self.font_size_spin.setRange(8, 24)

        # 确认按钮
        self.save_button = QtWidgets.QPushButton("保存设置", self)
        self.save_button.clicked.connect(self.save_settings)

        # 布局
        layout = QtWidgets.QFormLayout()
        layout.addRow(self.font_label, self.font_combo)
        layout.addRow(self.font_size_label, self.font_size_spin)
        layout.addWidget(self.save_button)

        self.setLayout(layout)
        self.load_settings()

    def load_settings(self):
        try:
            with open("settings.json", "r") as file:
                settings = json.load(file)
                self.font_combo.setCurrentFont(QtGui.QFont(settings["font"]))
                self.font_size_spin.setValue(settings["font_size"])
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def save_settings(self):
        settings = {
            "font": self.font_combo.currentFont().family(),
            "font_size": self.font_size_spin.value()
        }
        with open("settings.json", "w") as file:
            json.dump(settings, file)
        QtWidgets.QMessageBox.information(self, "保存成功", "设置已保存")
        self.accept()


class ReplacementApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_db()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle("替代料输入")

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

        # 添加SPEC选择文件按钮
        self.spec_button = QtWidgets.QPushButton("选择文件", self)
        self.spec_button.clicked.connect(self.select_spec_file)

        # 操作按钮
        self.add_button = QtWidgets.QPushButton("插入替代料", self)
        self.add_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.add_button.clicked.connect(self.add_replacement)

        self.no_replacement_button = QtWidgets.QPushButton("无替代料推荐", self)
        self.no_replacement_button.setStyleSheet("background-color: #2196F3; color: white;")
        self.no_replacement_button.clicked.connect(self.add_no_replacement)

        self.cancel_button = QtWidgets.QPushButton("取消", self)
        self.cancel_button.setStyleSheet("background-color: #f44336; color: white;")
        self.cancel_button.clicked.connect(self.close)

        self.settings_button = QtWidgets.QPushButton("设置", self)
        self.settings_button.clicked.connect(self.open_settings)

        # 提示标签
        self.message_label = QtWidgets.QLabel("")
        self.message_label.setAlignment(QtCore.Qt.AlignCenter)
        self.message_label.setObjectName("message_label")

        # 布局
        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(self.hhpn_label, self.hhpn_input)
        form_layout.addRow(self.aml_label, self.aml_input)
        form_layout.addRow(self.amlpn_label, self.amlpn_input)
        form_layout.addRow(self.isin_bom_label, self.isin_bom_input)
        form_layout.addRow(self.record_label, self.record_input)

        spec_layout = QtWidgets.QHBoxLayout()
        spec_layout.addWidget(self.spec_input)
        spec_layout.addWidget(self.spec_button)
        form_layout.addRow(self.spec_label, spec_layout)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.add_button)
        button_layout.addWidget(self.no_replacement_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.settings_button)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.message_label)

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

    def load_settings(self):
        try:
            with open("settings.json", "r") as file:
                settings = json.load(file)
                font = QtGui.QFont(settings["font"], settings["font_size"])
                self.setFont(font)
                # 应用按钮字体
                self.add_button.setFont(font)
                self.no_replacement_button.setFont(font)
                self.cancel_button.setFont(font)
                self.settings_button.setFont(font)
                # 设置窗口位置和大小
                self.resize(settings.get("width", 500), settings.get("height", 400))
                self.move(settings.get("x", 100), settings.get("y", 100))
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def add_replacement(self):
        hhpn = self.hhpn_input.text()
        aml = self.aml_input.text()
        amlpn = self.amlpn_input.text()
        isin_bom = self.isin_bom_input.text()
        record = self.record_input.text()
        spec = self.spec_input.text()
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            self.cursor.execute("INSERT INTO Relation (HHPN, AML, AMLPN, isin_bom, last_updated, record, SPEC) VALUES (?, ?, ?, ?, ?, ?, ?)",
                                (hhpn, aml, amlpn, isin_bom, last_updated, record, spec))
            self.conn.commit()
            self.show_message("插入成功", "green")
        except sqlite3.IntegrityError:
            self.show_message("数据已存在，插入失败", "red")

    def add_no_replacement(self):
        hhpn = self.hhpn_input.text()
        aml = self.aml_input.text()
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

    def select_spec_file(self):
        options = QtWidgets.QFileDialog.Options()
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择SPEC文件", "", "All Files (*);;Text Files (*.txt)", options=options)
        if file_path:
            file_name = file_path.split("/")[-1]
            self.spec_input.setText(file_name)

    def show_message(self, message, color):
        self.message_label.setText(message)
        self.message_label.setStyleSheet(f"color: {color};")
        QtCore.QTimer.singleShot(1000, self.clear_message)

    def clear_message(self):
        self.message_label.setText("")

    def open_settings(self):
        dialog = SettingsDialog()
        if dialog.exec_():
            self.load_settings()

    def closeEvent(self, event):
        self.conn.close()
        settings = {
            "font": self.font().family(),
            "font_size": self.font().pointSize(),
            "width": self.width(),
            "height": self.height(),
            "x": self.x(),
            "y": self.y()
        }
        with open("settings.json", "w") as file:
            json.dump(settings, file)

app = QtWidgets.QApplication(sys.argv)
window = ReplacementApp()
window.show()
sys.exit(app.exec_())

# Writing to
