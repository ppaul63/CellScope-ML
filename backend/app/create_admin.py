"""Create the first administrator account (or reset an admin's password).

Usage:
  python -m backend.app.create_admin --username admin
  echo "$PASSWORD" | python -m backend.app.create_admin --username admin --password-stdin
"""
import argparse
import getpass
import sys

from .auth import USERNAME_RE
from .db import connect, db_path, init_db
from .security import hash_password, password_problem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--username", required=True)
    ap.add_argument("--full-name", default="Administrator")
    ap.add_argument("--password-stdin", action="store_true", help="read the password from stdin instead of prompting")
    args = ap.parse_args()

    if not USERNAME_RE.match(args.username):
        sys.exit("Username must be 3-32 characters: letters, numbers, dot, dash or underscore.")
    if args.password_stdin:
        password = sys.stdin.readline().rstrip("\n")
    else:
        password = getpass.getpass("Password: ")
        if password != getpass.getpass("Repeat password: "):
            sys.exit("Passwords do not match.")
    if problem := password_problem(password):
        sys.exit(problem)

    init_db()
    conn = connect()
    try:
        row = conn.execute("SELECT id, role FROM users WHERE username = ?", (args.username,)).fetchone()
        if row and row["role"] != "admin":
            sys.exit(f"'{args.username}' exists as a regular user; choose another username.")
        if row:
            conn.execute("UPDATE users SET password_hash = ?, is_active = 1 WHERE id = ?", (hash_password(password), row["id"]))
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (row["id"],))
            action = "password reset for"
        else:
            conn.execute("INSERT INTO users (username, full_name, password_hash, role) VALUES (?, ?, ?, 'admin')",
                         (args.username, args.full_name, hash_password(password)))
            action = "created admin"
        conn.commit()
    finally:
        conn.close()
    print(f"{action} '{args.username}' in {db_path()}")


if __name__ == "__main__":
    main()
