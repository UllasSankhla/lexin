#!/usr/bin/env python3
"""
Add a new user to the Codelexin admin console.

Requires the Supabase service role key (Dashboard → Settings → API → service_role).

Usage:
    python add-user.py --email user@example.com --password SecurePass123

Environment variables (or set in .env):
    SUPABASE_URL           — e.g. https://xxxx.supabase.co
    SUPABASE_SERVICE_ROLE_KEY — the service role secret key
"""
import argparse
import os
import sys

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)


def create_user(email: str, password: str) -> None:
    supabase_url = os.environ.get("SUPABASE_URL", "https://tnuwpzjppoumdmqovxwh.supabase.co")
    service_role_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

    if not service_role_key:
        print("ERROR: SUPABASE_SERVICE_ROLE_KEY environment variable is not set.")
        print("Find it at: Supabase Dashboard → Settings → API → Project API keys → service_role")
        sys.exit(1)

    resp = requests.post(
        f"{supabase_url}/auth/v1/admin/users",
        json={"email": email, "password": password, "email_confirm": True},
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Content-Type": "application/json",
        },
        timeout=15,
    )

    if resp.status_code == 200:
        user = resp.json()
        print(f"✓ User created successfully")
        print(f"  Email:   {user['email']}")
        print(f"  User ID: {user['id']}")
        print(f"\nThe user can now sign in at the admin console.")
    else:
        print(f"ERROR: Failed to create user (HTTP {resp.status_code})")
        print(resp.json())
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add a new admin user to Codelexin")
    parser.add_argument("--email", required=True, help="User email address")
    parser.add_argument("--password", required=True, help="Initial password")
    args = parser.parse_args()

    create_user(args.email, args.password)
