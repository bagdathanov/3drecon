import sqlite3
import hashlib
import os

# Путь к БД (будет храниться в папке data/)
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "users.db")

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def initialize_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user'
        )
    ''')
    conn.commit()

    # 👤 Создание первого админа, если пользователей нет
    cursor.execute("SELECT COUNT(*) FROM users")
    count = cursor.fetchone()[0]
    if count == 0:
        admin_email = "admin@example.com"
        admin_password = hash_password("admin123")
        cursor.execute("INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
                       (admin_email, admin_password, "admin"))
        conn.commit()
        print(f"[INFO] Admin user created: {admin_email} / admin123")

    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def check_login(email: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password, role FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    if row and hash_password(password) == row["password"]:
        return {"id": row["id"], "role": row["role"]}
    return None



def register_user(email: str, password: str, role: str = "user") -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
            (email, hash_password(password), role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def check_login_with_role(email: str, password: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, password, role FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()

    if row and hash_password(password) == row["password"]:
        return {"id": row["id"], "role": row["role"]}
    return None


# При запуске файла напрямую
if __name__ == "__main__":
    initialize_db()
    print("Database initialized at:", DB_PATH)