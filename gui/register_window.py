from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox
from core.auth import register_user

class RegisterWindow(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Регистрация")
        self.setFixedSize(300, 200)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout()

        self.label_email = QLabel("Email:")
        self.input_email = QLineEdit()

        self.label_password = QLabel("Пароль:")
        self.input_password = QLineEdit()
        self.input_password.setEchoMode(QLineEdit.Password)

        self.label_confirm = QLabel("Подтвердите пароль:")
        self.input_confirm = QLineEdit()
        self.input_confirm.setEchoMode(QLineEdit.Password)

        self.btn_register = QPushButton("Зарегистрироваться")
        self.btn_register.clicked.connect(self.handle_register)

        layout.addWidget(self.label_email)
        layout.addWidget(self.input_email)
        layout.addWidget(self.label_password)
        layout.addWidget(self.input_password)
        layout.addWidget(self.label_confirm)
        layout.addWidget(self.input_confirm)
        layout.addWidget(self.btn_register)

        self.setLayout(layout)

    def handle_register(self):
        email = self.input_email.text().strip()
        password = self.input_password.text().strip()
        confirm = self.input_confirm.text().strip()

        if not email or not password:
            QMessageBox.warning(self, "Ошибка", "Введите email и пароль")
            return

        if password != confirm:
            QMessageBox.warning(self, "Ошибка", "Пароли не совпадают")
            return

        if register_user(email, password):
            QMessageBox.information(self, "Успех", "Регистрация прошла успешно")
            self.accept()
        else:
            QMessageBox.warning(self, "Ошибка", "Пользователь уже существует")