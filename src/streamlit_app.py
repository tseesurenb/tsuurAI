import streamlit as st

st.set_page_config(page_title="GPU Server App", page_icon="🚀", layout="wide")

st.title("🚀 GPU Server Dashboard")
st.write("Running on remote GPU server")

col1, col2 = st.columns(2)

with col1:
    st.header("System Info")
    if st.button("Check GPU"):
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            st.code(result.stdout)
        except:
            st.warning("nvidia-smi not available")

with col2:
    st.header("Quick Test")
    name = st.text_input("Enter your name")
    if name:
        st.success(f"Hello, {name}! 👋")

    if st.button("Celebrate 🎉"):
        st.balloons()

st.divider()
st.caption("Streamlit app running on 160.187.40.172")
