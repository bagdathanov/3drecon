import sys
import multiprocessing
from PyQt5.QtWidgets import QApplication
from gui.main_window import MainWindow
from gui.login_window import LoginWindow
from core.auth import initialize_db  

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn') 
    app = QApplication(sys.argv)

    initialize_db()  

    login_window = LoginWindow()
    if login_window.exec_() == login_window.Accepted:
        main_window = MainWindow(user_id=login_window.user_id, user_role=login_window.user_role)
        main_window.show()
        sys.exit(app.exec())
