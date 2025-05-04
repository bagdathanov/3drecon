from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
)
from gui.register_window import RegisterWindow
from core.auth import check_login, check_login_with_role

class LoginWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Вход в систему")
        self.setFixedSize(300, 220)
        self.user_id = None
        self.user_role = "user"  # 👈 по умолчанию
        self.setup_ui()


    def setup_ui(self):
        layout = QVBoxLayout()

        self.label_email = QLabel("Email:")
        self.input_email = QLineEdit()

        self.label_password = QLabel("Пароль:")
        self.input_password = QLineEdit()
        self.input_password.setEchoMode(QLineEdit.Password)

        self.btn_login = QPushButton("Войти")
        self.btn_login.clicked.connect(self.handle_login)

        self.btn_register = QPushButton("Регистрация")
        self.btn_register.clicked.connect(self.open_register)

        layout.addWidget(self.label_email)
        layout.addWidget(self.input_email)
        layout.addWidget(self.label_password)
        layout.addWidget(self.input_password)
        layout.addWidget(self.btn_login)
        layout.addWidget(self.btn_register)

        self.setLayout(layout)

    def handle_login(self):
        email = self.input_email.text().strip()
        password = self.input_password.text().strip()

        if not email or not password:
            QMessageBox.warning(self, "Ошибка", "Введите email и пароль")
            return

        user = check_login_with_role(email, password)
        if user:
            self.user_id = user["id"]
            self.user_role = user["role"]  # ✅ сохраняем роль
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Неверный email или пароль")

    def open_register(self):
        reg_win = RegisterWindow()
        if reg_win.exec_() == reg_win.Accepted:
            QMessageBox.information(self, "Готово", "Теперь вы можете войти с новым аккаунтом.")
