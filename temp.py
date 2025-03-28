import streamlit as st
import random
import string
import io

def generate_random_text(length=1000, include_numbers=True, include_special=False):
    """Generate random text with specified parameters"""
    chars = string.ascii_letters
    if include_numbers:
        chars += string.digits
    if include_special:
        chars += string.punctuation
    
    return ''.join(random.choice(chars) for _ in range(length))

# Set page title
st.title("Random Text File Generator")
st.write("Generate random text files and download them with one click!")

# Sidebar options for customization
st.sidebar.header("Customization Options")
text_length = st.sidebar.slider("Text Length", 100, 10000, 1000)
include_numbers = st.sidebar.checkbox("Include Numbers", value=True)
include_special = st.sidebar.checkbox("Include Special Characters", value=False)
filename = st.sidebar.text_input("Filename", value="random_text.txt")

# Ensure filename has .txt extension
if not filename.endswith('.txt'):
    filename += '.txt'

# Generate random text button
if st.button("Generate Random Text"):
    random_text = generate_random_text(
        length=text_length,
        include_numbers=include_numbers,
        include_special=include_special
    )
    
    # Display preview
    st.subheader("Preview:")
    st.text_area("Random Text Preview", random_text[:500] + 
                ("..." if len(random_text) > 500 else ""), 
                height=200)
    
    # Create download button
    st.download_button(
        label="Download Text File",
        data=random_text,
        file_name=filename,
        mime="text/plain"
    )
    
    # Show file info
    st.info(f"File size: {len(random_text)/1024:.2f} KB")

# Add some helpful information
st.markdown("---")
st.markdown("""
### How it works:
1. Choose your customization options in the sidebar
2. Click 'Generate Random Text' button
3. Preview the generated text
4. Download your file with the 'Download Text File' button
""")