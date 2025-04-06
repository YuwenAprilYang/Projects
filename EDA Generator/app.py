# Import packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Page Header
st.title("EDA Generator")
st.write("Upload a CSV file to generate an interactive exploratory data analysis report.")
st.write("Or click the button below to load an example file.")


# Initialize session state variables

# Remember which column was last selected for distribution plots
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None

# Flag to track whether to use the example dataset
if "use_example" not in st.session_state:
    st.session_state.use_example = False

# Store the currently loaded DataFrame
if "df" not in st.session_state:
    st.session_state.df = None

# Section
if "active_section" not in st.session_state:
    st.session_state.active_section = "Preview"

# Caching functions to load data
@st.cache_data
def load_csv_data(file_obj):
    try:
        return pd.read_csv(file_obj)
    except Exception as e:
        st.error(f"Error reading the uploaded file: {e}")
        return None

@st.cache_data
def load_example_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading the example file: {e}")
        return None
    
# File uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type = "csv")
example_url = "https://raw.githubusercontent.com/YuwenAprilYang/Projects/a27c0a1a91b0dc784a1cef0e2b0344c4a0cd15f3/EDA%20Generator/Example%20file.csv"

# If the user uploads a file, load it
if uploaded_file is not None:
    st.session_state.df = load_csv_data(uploaded_file)
    st.session_state.use_example = False

# If the user clicks “Use Example”, set the flag and load example data
if st.button("Click to Use Example Dataset"):
    st.session_state.use_example = True
    st.session_state.df = load_example_data(example_url)

# If flag is set (and no upload), ensure df is loaded
if st.session_state.use_example and st.session_state.df is None:
    st.session_state.df = load_example_data(example_url)

# Assign DataFrame from session state
df = st.session_state.df

# Main EDA Interface

if df is not None:
    try:
        
        # Create tabs for different sections
        tabs = st.tabs([
            "Preview",
            "Info",
            "Descriptive Stats",
            "Correlation",
            "Distribution",
        ])
        
        # Data preview
        with tabs[0]:
            st.header("Data Preview")
            st.subheader("First 10 Rows")
            st.dataframe(df.head(10))
            st.subheader("Last 10 Rows")
            st.dataframe(df.tail(10))

        # Basic information
        with tabs[1]:
            st.header("Dataset Information")
            st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
            st.write("**Columns:**", df.columns.tolist())

            # Data types
            st.subheader("Data Types")
            dtypes = pd.DataFrame(df.dtypes,columns=["Data Type"])
            st.dataframe(dtypes)

            # Missing Values
            st.subheader("Missing Values")
            missing_values = pd.DataFrame(df.isnull().sum(),columns = ["Missing Values"])
            st.dataframe(missing_values)
        

        # Descriptive Statistics
        with tabs[2]:
            st.header("Descriptive Statistics")
            st.dataframe(df.describe())

        # Correlation Heatmap
        with tabs[3]:
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            if numeric_columns:
                st.header("Correlation Heatmap")
                plt.figure(figsize=(10,8))
                corr = df[numeric_columns].corr()
                sns.heatmap(corr,annot=True, cmap="coolwarm",fmt='.2f')
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                st.info("No numeric columns to display.")
        
        # Distribution plot for a selected numeric column
        with tabs[4]:
            if numeric_columns:
                st.header("Distribution Plot")
                # Preserve the selected column in session state
                selected_idx = numeric_columns.index(st.session_state.selected_column) if st.session_state.selected_column in numeric_columns else 0
                col_to_plot = st.selectbox("Select a numeric column for distribution plot", options=numeric_columns, index=selected_idx)
                st.session_state.selected_column = col_to_plot  
                if col_to_plot:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(df[col_to_plot].dropna(), kde=True)
                    plt.title(f"Distribution of {col_to_plot}")
                    st.pyplot(plt.gcf())
                    plt.clf()
                else:
                    st.info("No numeric columns to plo.")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file to get started.")

