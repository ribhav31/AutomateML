# import streamlit as st
# import pandas as pd
# import os
# from pandas_profiling import ProfileReport
# from streamlit_pandas_profiling import st_profile_report
# from pycaret.classification import setup, compare_models, pull, save_model

# # Function to download trained model
# def get_binary_file_downloader_html(bin_file, label='Download'):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     bin_str = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{label}</a>'
#     return href

# st.set_page_config(page_title="AdvancedML", page_icon="🧠")

# with st.sidebar:
#     st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
#     st.title("AdvancedML")
#     choice = st.radio("Navigation", ["Uploading", "Profiling", "ML", "Download"])

#     st.info("This application allows you to build an automated machine learning pipeline using Streamlit, pandas, pandas_profiling, pycaret, and works like magic!!")

# if os.path.exists("SourceData.csv"):
#     df = pd.read_csv("SourceData.csv")

# if choice == "Uploading":
#     st.title("Upload your data for Custom Modelling!!")
#     file = st.file_uploader("Upload your file here")
#     if file:
#         df = pd.read_csv(file)
#         df.to_csv("SourceData.csv", index=None)
#         st.dataframe(df)

# if choice == "Profiling":
#     st.title("Exploratory Data Analysis")
#     if st.button("Generate Profile Report"):
#         profile = ProfileReport(df, explorative=True)
#         profile.to_file("profile_report.html")
#         st_profile_report(profile)

# if choice == "ML":
#     st.title("Machine Learning Model Working!!")
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'):
#         setup_df = setup(df, target=chosen_target)
#         st.dataframe(setup_df)
#         best_model = compare_models()
#         compare_df = pull()
#         st.dataframe(compare_df)
#         save_model(best_model, 'best_model')

# if choice == "Download":
#     with st.spinner("Downloading..."):
#         st.markdown(
#             get_binary_file_downloader_html("trained_model.pkl", "Download trained_model.pkl"),
#             unsafe_allow_html=True,
#         )

import streamlit as st
import pandas as pd
import os
import base64
from pandas_profiling import ProfileReport
import pandas as pd
import streamlit as st
import os
import base64
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import base64
import os
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import base64
import os

import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
import base64
import os

# # Function to download trained model
# def get_binary_file_downloader_html(bin_file, label='Download'):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     bin_str = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{label}</a>'
#     return href

# st.set_page_config(page_title="AdvancedML", page_icon="🧠")

# with st.sidebar:
#     st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
#     st.title("AdvancedML")
#     choice = st.radio("Navigation", ["Uploading", "Profiling", "ML", "Download"])

#     st.info("This application allows you to build an automated machine learning pipeline using Streamlit, pandas, and works like magic!!")

# if os.path.exists("SourceData.csv"):
#     df = pd.read_csv("SourceData.csv")

# if choice == "Uploading":
#     st.title("Upload your data for Custom Modelling!!")
#     file = st.file_uploader("Upload your file here")
#     if file:
#         df = pd.read_csv(file)
#         df.to_csv("SourceData.csv", index=None)
#         st.dataframe(df)

# if choice == "Profiling":
#     st.title("Exploratory Data Analysis")
    
#     # Display summary statistics
#     st.write("### Summary Statistics:")
#     st.write(df.describe())
    
#     # Display correlation matrix for numeric columns
#     numeric_columns = df.select_dtypes(include=['number']).columns
#     corr_matrix = df[numeric_columns].corr()
    
#     st.write("### Correlation Matrix:")
#     st.write(corr_matrix)
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     st.pyplot()

#     # Display pair plot
#     st.write("### Pair Plot:")
#     sns.pairplot(df)
#     st.pyplot()

# if choice == "ML":
#     st.title("Machine Learning Model Working!!")
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'):
#         # Preprocess the data
#         X = df.drop(chosen_target, axis=1)
#         y = df[chosen_target]

#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         for column in X.select_dtypes(include='object').columns:
#             X[column] = label_encoder.fit_transform(X[column])

#         # Impute missing values
#         imputer = SimpleImputer(strategy='mean')
#         X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         model = RandomForestClassifier()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         st.write("Classification Report:")
#         st.text(classification_report(y_test, y_pred))

#         st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

#         # Save the model (example, you might want to use a better method for your use case)
#         with open("trained_model.pkl", "wb") as model_file:
#             pickle.dump(model, model_file)

#         st.success("Modeling completed successfully!")

# if choice == "Download":
#     with st.spinner("Downloading..."):
#         st.markdown(
#             get_binary_file_downloader_html("trained_model.pkl", "Download trained_model.pkl"),
#             unsafe_allow_html=True,
#         )


# import streamlit as st
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer
# import pickle
# import base64
# import os

# # Function to download trained model
# def get_binary_file_downloader_html(bin_file, label='Download'):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     bin_str = base64.b64encode(data).decode()
#     href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{label}</a>'
#     return href

# st.set_page_config(page_title="AdvancedML", page_icon="🧠")

# with st.sidebar:
#     st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
#     st.title("AdvancedML")
#     choice = st.radio("Navigation", ["Uploading", "Profiling", "ML", "Download"])

#     st.info("This application allows you to build an automated machine learning pipeline using Streamlit, pandas, and works like magic!!")

# if os.path.exists("SourceData.csv"):
#     df = pd.read_csv("SourceData.csv")

# if choice == "Uploading":
#     st.title("Upload your data for Custom Modelling!!")
#     file = st.file_uploader("Upload your file here")
#     if file:
#         df = pd.read_csv(file)
#         df.to_csv("SourceData.csv", index=None)
#         st.dataframe(df)

# if choice == "Profiling":
#     st.title("Exploratory Data Analysis")
    
#     # Display summary statistics
#     st.write("### Summary Statistics:")
#     st.write(df.describe())
    
#     # Display correlation matrix for numeric columns
#     numeric_columns = df.select_dtypes(include=['number']).columns
#     corr_matrix = df[numeric_columns].corr()
    
#     st.write("### Correlation Matrix:")
#     st.write(corr_matrix)
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
#     st.pyplot()

#     # Display pair plot
#     st.write("### Pair Plot:")
#     pair_plot = sns.pairplot(df, corner=True)  # Set corner=True to fix the numpy boolean subtract error
#     st.pyplot(pair_plot.fig)

# if choice == "ML":
#     st.title("Machine Learning Model Working!!")
#     chosen_target = st.selectbox('Choose the Target Column', df.columns)
#     if st.button('Run Modelling'):
#         # Preprocess the data
#         X = df.drop(chosen_target, axis=1)
#         y = df[chosen_target]

#         # Encode categorical variables
#         label_encoder = LabelEncoder()
#         for column in X.select_dtypes(include='object').columns:
#             X[column] = label_encoder.fit_transform(X[column])

#         # Impute missing values
#         imputer = SimpleImputer(strategy='mean')
#         X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         model = RandomForestClassifier()
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         st.write("Classification Report:")
#         st.text(classification_report(y_test, y_pred))

#         st.write("Accuracy Score:", accuracy_score(y_test, y_pred))

#         # Save the model (example, you might want to use a better method for your use case)
#         with open("trained_model.pkl", "wb") as model_file:
#             pickle.dump(model, model_file)

#         st.success("Modeling completed successfully!")

# if choice == "Download":
#     with st.spinner("Downloading..."):
#         st.markdown(
#             get_binary_file_downloader_html("trained_model.pkl", "Download trained_model.pkl"),
#             unsafe_allow_html=True,
#         )



import streamlit as st
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification
import pickle
import base64
import os

# Function to download trained model
def get_binary_file_downloader_html(bin_file, label='Download'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{label}</a>'
    return href

def data_cleaning(df, clean_option, numeric_clean_option):
    # Clean missing values
    if clean_option == "Rows":
        df = df.dropna(axis=0)
    elif clean_option == "Columns":
        df = df.dropna(axis=1)

    # Handle missing values for numeric columns
    if numeric_clean_option == "Fill with Mean":
        numeric_columns = df.select_dtypes(include=['number']).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    elif numeric_clean_option == "Drop":
        df = df.dropna(subset=df.select_dtypes(include=['number']).columns, axis=0)

    return df

# Function to create a synthetic feature
def create_synthetic_feature(samples, informative_features, redundant_features, random_state):
    X, y = make_classification(
        n_samples=samples,
        n_features=1 + informative_features + redundant_features,
        n_informative=informative_features,
        n_redundant=redundant_features,
        random_state=random_state
    )
    synthetic_feature = pd.DataFrame(X[:, 0], columns=['SyntheticFeature'])
    return synthetic_feature

# Function to define a simple neural network model
def create_neural_network(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

st.set_page_config(page_title="AdvancedML", page_icon="🧠")

with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("AdvancedML")
    choice = st.radio("Navigation", ["Uploading", "Profiling", "Data Cleaning", "ML", "Download", "New Feature", "Data Splitting"])

    st.info("This application allows you to build an automated machine learning pipeline using Streamlit, pandas, and works like magic!!")

if os.path.exists("SourceData.csv"):
    df = pd.read_csv("SourceData.csv")

if choice == "Uploading":
    st.title("Upload your data for Custom Modelling!!")
    file = st.file_uploader("Upload your file here")
    if file:
        df = pd.read_csv(file)
        df.to_csv("SourceData.csv", index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    
    # Display summary statistics
    st.write("### Summary Statistics:")
    st.write(df.describe())
    
    # Display correlation matrix for numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns
    corr_matrix = df[numeric_columns].corr()
    
    st.write("### Correlation Matrix:")
    st.write(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

    # Display pair plot
    st.write("### Pair Plot:")
    pair_plot = sns.pairplot(df, corner=True)  # Set corner=True to fix the numpy boolean subtract error
    st.pyplot(pair_plot.fig)

if choice == "Data Cleaning":
    st.title("Data Cleaning")
    st.write("### Original Data:")
    st.dataframe(df)

    # Data cleaning options
    clean_option = st.selectbox("Select cleaning option", ["None", "Rows", "Columns"])
    numeric_clean_option = "None"

    if clean_option != "None" and df.select_dtypes(include=['number']).columns.any():
        numeric_clean_option = st.selectbox("Select numeric cleaning option", ["None", "Fill with Mean", "Drop"])

    if st.button('Clean Data'):
        df_cleaned = data_cleaning(df, clean_option, numeric_clean_option)
        st.write("### Cleaned Data:")
        st.dataframe(df_cleaned)

if choice == "ML":
    st.title("Machine Learning Model Working!!")
    chosen_target = st.selectbox('Choose the Target Column', df.columns)
    if st.button('Run Modelling'):
        # Preprocess the data
        X = df.drop(chosen_target, axis=1)
        y = df[chosen_target]

        # Encode categorical variables
        label_encoder = LabelEncoder()
        for column in X.select_dtypes(include='object').columns:
            X[column] = label_encoder.fit_transform(X[column])

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Use RandomForestClassifier for demonstration
        model_rf = RandomForestClassifier()
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)

        st.write("RandomForest Classification Report:")
        st.text(classification_report(y_test, y_pred_rf))

        st.write("RandomForest Accuracy Score:", accuracy_score(y_test, y_pred_rf))

        # Save the RandomForest model (example, you might want to use a better method for your use case)
        with open("trained_model_rf.pkl", "wb") as model_file_rf:
            pickle.dump(model_rf, model_file_rf)

        # Use Neural Network for demonstration
        model_nn = create_neural_network(input_dim=X_train.shape[1])
        model_nn.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        y_pred_nn = model_nn.predict(X_test)
        y_pred_nn = (y_pred_nn > 0.5).astype(int)

        st.write("Neural Network Classification Report:")
        st.text(classification_report(y_test, y_pred_nn))

        st.write("Neural Network Accuracy Score:", accuracy_score(y_test, y_pred_nn))

        # Save the Neural Network model (example, you might want to use a better method for your use case)
        model_nn.save("trained_model_nn.h5")

        st.success("Modeling completed successfully!")

if choice == "Download":
    with st.spinner("Downloading..."):
        st.markdown(
            get_binary_file_downloader_html("trained_model_rf.pkl", "Download trained_model_rf.pkl"),
            unsafe_allow_html=True,
        )

        st.markdown(
            get_binary_file_downloader_html("trained_model_nn.h5", "Download trained_model_nn.h5"),
            unsafe_allow_html=True,
        )

# New Feature
if choice == "New Feature":
    st.title("Create a New Synthetic Feature")
    st.info("Generate a synthetic feature using scikit-learn's make_classification function.")

    samples = st.slider("Number of Samples", min_value=10, max_value=1000, value=100)
    informative_features = st.slider("Number of Informative Features", min_value=1, max_value=10, value=5)
    redundant_features = st.slider("Number of Redundant Features", min_value=0, max_value=5, value=2)
    random_state = st.number_input("Random State", value=42)

    synthetic_feature = create_synthetic_feature(samples, informative_features, redundant_features, random_state)
    st.write("### Synthetic Feature:")
    st.dataframe(synthetic_feature)

    # Add synthetic feature to the dataset
    if st.button("Add Synthetic Feature to Dataset"):
        df = pd.concat([df, synthetic_feature], axis=1)
        st.success("Synthetic feature added to the dataset!")

# Data Splitting
if choice == "Data Splitting":
    st.title("Data Splitting")
    st.write("### Original Dataset:")
    st.dataframe(df)

    # Target variable selection
    chosen_target_splitting = st.selectbox('Choose the Target Column for Data Splitting', df.columns)

    # Data splitting options
    test_size = st.slider("Test Set Size", min_value=0.1, max_value=0.5, step=0.1, value=0.2, format="%f")
    
    if st.button('Split Data'):
        # Perform data splitting
        X_train, X_test, y_train, y_test = train_test_split(df.drop(chosen_target_splitting, axis=1), df[chosen_target_splitting], test_size=test_size, random_state=42)

        st.write("### Training Set:")
        st.dataframe(X_train)
        st.write("### Testing Set:")
        st.dataframe(X_test)

# Display the updated dataset
st.write("### Updated Dataset:")
st.dataframe(df)


