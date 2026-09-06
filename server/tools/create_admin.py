from __future__ import annotations
import argparse
import sqlalchemy
from app.core.database import SessionLocal, init_auth_database
from app.core.security import hash_password
from app.models.user import User, UserRole

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update an admin user")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", required=True)
    parser.add_argument("--role", default="admin")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if len(args.password) < 12:
        raise SystemExit("--password must be at least 12 characters")
    init_auth_database()

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == args.username).first()
        
        try:
            role_enum = UserRole(args.role.lower())
        except ValueError:
            role_enum = UserRole.admin if args.role.lower() == "admin" else UserRole.operator

        if user is None:
            # Lấy ID lớn nhất hiện tại để sinh ID tiếp theo (dạng 00000X)
            # Hoặc để Postgres tự sinh nếu bạn đã thiết lập DEFAULT LPAD...
            user = User(
                username=args.username,
                password_hash=hash_password(args.password),
                role=role_enum,
                full_name="System Administrator" if role_enum == UserRole.admin else "Operator"
            )
            db.add(user)
            action = "created"
        else:
            # Nếu user đã tồn tại, chỉ cập nhật mật khẩu và role
            user.password_hash = hash_password(args.password)
            user.role = role_enum
            action = "updated"

        db.commit()
        print(f"User '{args.username}' {action} successfully with correct bcrypt hash.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
