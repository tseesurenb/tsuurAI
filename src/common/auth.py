"""
TsuurAI Authentication Module
Supports: Local registration/login and Google OAuth
"""

import streamlit as st
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime
import bcrypt

from .config import DATA_DIR

# User database file
USERS_FILE = DATA_DIR / "users.json"
USAGE_FILE = DATA_DIR / "usage.json"

def init_data_dir():
    """Initialize data directory"""
    DATA_DIR.mkdir(exist_ok=True)

    if not USERS_FILE.exists():
        USERS_FILE.write_text(json.dumps({"users": {}}, indent=2))

    if not USAGE_FILE.exists():
        USAGE_FILE.write_text(json.dumps({"usage": []}, indent=2))

def load_users():
    """Load users from JSON file"""
    init_data_dir()
    return json.loads(USERS_FILE.read_text())

def save_users(data):
    """Save users to JSON file"""
    init_data_dir()
    USERS_FILE.write_text(json.dumps(data, indent=2))

def hash_password(password):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    """Verify password against hash"""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(email, name, password):
    """Register a new user"""
    data = load_users()

    if email in data["users"]:
        return False, "Email already registered"

    data["users"][email] = {
        "name": name,
        "password": hash_password(password),
        "created_at": datetime.now().isoformat(),
        "auth_type": "local",
        "usage_count": 0
    }

    save_users(data)
    return True, "Registration successful"

def login_user(email, password):
    """Login with email and password"""
    data = load_users()

    if email not in data["users"]:
        return False, None, "Email not found"

    user = data["users"][email]

    if user.get("auth_type") == "google":
        return False, None, "Please use Google Sign-In for this account"

    if not verify_password(password, user["password"]):
        return False, None, "Incorrect password"

    return True, user, "Login successful"

def register_google_user(email, name):
    """Register or login a Google user"""
    data = load_users()

    if email not in data["users"]:
        # New Google user
        data["users"][email] = {
            "name": name,
            "password": None,
            "created_at": datetime.now().isoformat(),
            "auth_type": "google",
            "usage_count": 0
        }
        save_users(data)

    return data["users"][email]

def log_usage(email, action, details=None):
    """Log user activity"""
    init_data_dir()
    data = json.loads(USAGE_FILE.read_text())

    data["usage"].append({
        "email": email,
        "action": action,
        "details": details,
        "timestamp": datetime.now().isoformat()
    })

    # Keep only last 1000 entries
    data["usage"] = data["usage"][-1000:]

    USAGE_FILE.write_text(json.dumps(data, indent=2))

    # Update user's usage count
    users_data = load_users()
    if email in users_data["users"]:
        users_data["users"][email]["usage_count"] = users_data["users"][email].get("usage_count", 0) + 1
        users_data["users"][email]["last_active"] = datetime.now().isoformat()
        save_users(users_data)

def get_user_stats(email):
    """Get user statistics"""
    data = load_users()
    if email in data["users"]:
        user = data["users"][email]
        return {
            "name": user["name"],
            "usage_count": user.get("usage_count", 0),
            "created_at": user.get("created_at", "Unknown"),
            "last_active": user.get("last_active", "Never")
        }
    return None

def show_login_page():
    """Display login/registration page"""

    st.title("TsuurAI - Speech to Text")
    st.subheader("Please sign in to continue")

    # Check if already logged in
    if st.session_state.get("authenticated"):
        return True

    tab1, tab2 = st.tabs(["Sign In", "Register"])

    with tab1:
        st.markdown("### Sign In")

        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign In", type="primary")

            if submit:
                if email and password:
                    success, user, message = login_user(email, password)
                    if success:
                        st.session_state["authenticated"] = True
                        st.session_state["user_email"] = email
                        st.session_state["user_name"] = user["name"]
                        log_usage(email, "login")
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please enter email and password")

        st.divider()

        # Google Sign-In placeholder
        st.markdown("### Or sign in with Google")

        # Check for Google OAuth environment variables
        google_client_id = os.environ.get("GOOGLE_CLIENT_ID")

        if google_client_id:
            if st.button("Sign in with Google", type="secondary", use_container_width=True):
                st.info("Google OAuth - See setup instructions below")
        else:
            st.info("Google Sign-In not configured. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")

            with st.expander("Google OAuth Setup Instructions"):
                st.markdown("""
                1. Go to [Google Cloud Console](https://console.cloud.google.com/)
                2. Create a new project or select existing
                3. Enable "Google+ API" or "Google Identity"
                4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
                5. Application type: "Web application"
                6. Add authorized redirect URI: `https://your-server:8501`
                7. Copy Client ID and Client Secret
                8. Set environment variables:
                ```bash
                export GOOGLE_CLIENT_ID="your-client-id"
                export GOOGLE_CLIENT_SECRET="your-client-secret"
                ```
                """)

    with tab2:
        st.markdown("### Create Account")

        with st.form("register_form"):
            reg_name = st.text_input("Full Name")
            reg_email = st.text_input("Email Address")
            reg_password = st.text_input("Password", type="password")
            reg_password2 = st.text_input("Confirm Password", type="password")
            reg_submit = st.form_submit_button("Create Account", type="primary")

            if reg_submit:
                if not all([reg_name, reg_email, reg_password, reg_password2]):
                    st.error("Please fill in all fields")
                elif reg_password != reg_password2:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif "@" not in reg_email:
                    st.error("Please enter a valid email")
                else:
                    success, message = register_user(reg_email, reg_name, reg_password)
                    if success:
                        st.success(message + " Please sign in.")
                    else:
                        st.error(message)

    return False

def show_user_sidebar():
    """Show user info in sidebar"""
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.divider()
            st.markdown(f"**Logged in as:** {st.session_state.get('user_name', 'User')}")
            st.caption(st.session_state.get('user_email', ''))

            # User stats
            stats = get_user_stats(st.session_state.get('user_email'))
            if stats:
                st.caption(f"Transcriptions: {stats['usage_count']}")

            if st.button("Sign Out", use_container_width=True):
                log_usage(st.session_state.get('user_email'), "logout")
                st.session_state["authenticated"] = False
                st.session_state["user_email"] = None
                st.session_state["user_name"] = None
                st.rerun()

def require_auth(func):
    """Decorator to require authentication"""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated"):
            show_login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper
