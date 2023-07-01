import streamlit as st
import pickle
import base64
from streamlit_option_m
enu import option_menu
with open("model.pkl","rb") as file:
    model = pickle.load(file)
st.set_page_config(page_icon='',layout='wide',initial_sidebar_state='expanded')
with st.sidebar:
    option = option_menu("Main menu",["Home","About","EDA Report","Application"],icons=['house','database',"",'gear'], menu_icon="cast",orientation="vertical",default_index=3)
if option == "Home":
    st.image("Images/home.png",use_column_width=True)
if option == "About":
    st.markdown("<h5>About Project</h5>", unsafe_allow_html=True)
    about_data = """
            **Summary**: Breast cancer prediction using machine learning (ML) is of utmost importance due to its potential to enhance early detection and improve treatment outcomes. ML algorithms can analyze extensive patient data, identifying intricate patterns and risk factors that may elude conventional approaches. By leveraging this technology, healthcare professionals can accurately predict breast cancer, empowering them to intervene at earlier stages of the disease. Timely intervention can lead to better treatment strategies, potentially saving lives and improving patient outcomes. ML also enables personalized medicine, tailoring treatment plans to individual patients based on their unique characteristics. Overall, ML-based breast cancer prediction holds immense promise in advancing healthcare by optimizing diagnosis, treatment, and patient care.

            **Background**: This dataset consists of a group of breast cancer patients, who had surgery to remove their tumour. The dataset consists of the following variables - 

            *Patient_ID: unique identifier id of a patient*

            *Age: age at diagnosis (Years)*

            *Gender: Male/Female*

            Protein1, Protein2, Protein3, Protein4: expression levels (undefined units)

            *Tumour_Stage: I, II, III*

            *Histology: Infiltrating Ductal Carcinoma, Infiltrating Lobular Carcinoma, Mucinous Carcinoma*

            *ER status: Positive/Negative*

            *PR status: Positive/Negative*

            *HER2 status: Positive/Negative*

            *Surgery_type: Lumpectomy, Simple Mastectomy, Modified Radical Mastectomy, Other*

            *Date_of_Surgery: Date on which surgery was performed (in DD-MON-YY)*

            *Date_of_Last_Visit: Date of last visit (in DD-MON-YY) [can be null, in case the patient didn’t visited again after the surgery]*

            *Patient_Status: Alive/Dead [can be null, in case the patient didn’t visited again after the surgery and there is no information available whether the patient is alive or dead].*
                """
    st.markdown(about_data)
    about_model = """
            **Model**: Support Vector Machine 
            Support Vector Machines (SVM) is a powerful machine learning algorithm used for classification and regression tasks. SVM aims to find the best hyperplane that separates data points of different classes in a high-dimensional space. It identifies a decision boundary that maximizes the margin between classes, allowing for better generalization and improved prediction accuracy. SVM can handle both linearly separable and non-linearly separable data by utilizing kernel functions to map the input data into a higher-dimensional feature space. This enables the algorithm to capture complex relationships between variables.

            For more details on SVM implementation and usage, you can refer to the scikit-learn library in Python, which provides a comprehensive implementation of SVM with various kernels and customizable parameters. The scikit-learn documentation provides detailed explanations, examples, and code snippets for SVM:

            https://scikit-learn.org/stable/modules/svm.html

            By exploring the documentation, you can gain a deeper understanding of SVM and how to apply it to your specific classification or regression problem.
                """
    st.markdown(about_model)
if option == "EDA Report":
    st.markdown("<h5>Exploratory Data Analysis</h5>", unsafe_allow_html=True)
    content = """Exploratory Data Analysis (EDA) is a crucial step in the data analysis process. It involves examining and understanding the data to uncover patterns, identify anomalies, and gain insights before applying any modeling or statistical techniques. EDA encompasses various tasks such as data cleaning, data visualization, and summary statistics. 

During EDA, data quality issues like missing values, outliers, or inconsistencies are addressed through techniques like imputation or removal. Visualizations such as histograms, scatter plots, and box plots help to understand the distribution, relationships, and trends in the data. Summary statistics provide measures of central tendency, dispersion, and correlation.

EDA helps to discover patterns, relationships, and potential challenges in the data, which inform subsequent analysis decisions. It aids in feature selection, model validation, and hypothesis generation. EDA is often an iterative process, where initial insights lead to further investigation and refinement of analysis techniques.

By conducting EDA, data analysts and scientists can gain a deeper understanding of the data, identify interesting patterns, and make informed decisions about the appropriate modeling techniques or further data collection efforts."""
    st.markdown(content)
    # Read the HTML content from the selected document
    with open("Report.html", "r") as file:
        html_content = file.read()
    # Create a download link for the HTML document
    b64_html = base64.b64encode(html_content.encode()).decode()
    download_link = f'<a href="data:text/html;base64,{b64_html}" download="document.html">Download report & Open document to view in browser</a>'

    # Display the download link
    st.markdown(download_link, unsafe_allow_html=True)
if option == "Application":
    st.markdown("<h1>Prediction App</h1",unsafe_allow_html=True)
    cols = st.columns(2)
    Age = cols[0].number_input("Enter age of patient")
    gender = {"MALE": 0, "FEMALE": 1}
    Gender = cols[1].selectbox('Select Gender',list(gender.keys()))
    protein_1 = cols[0].number_input("Enter value of Protein 1")
    protein_2 = cols[1].number_input("Enter value of Protein 2")
    protein_3 = cols[0].number_input("Enter value of Protein 3")
    protein_4 = cols[1].number_input("Enter value of Protein 4")
    tumorstage = {"I": 1, "II": 2, "III": 3}
    Tumorstage = cols[0].selectbox('Select Tumor stage',list(tumorstage.keys()))
    histology = {"Infiltrating Ductal Carcinoma": 1,
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3}
    Histology = cols[1].selectbox('Histology',list(histology.keys()))
    erstatus = {"Positive": 1}
    ERstatus = cols[0].selectbox("ER status",list(erstatus.keys()))
    prstatus = {"Positive": 1}
    PRstatus = cols[1].selectbox("PR status", list(prstatus.keys()))
    her2status = {"Positive": 1,"Negative":2}
    HER2status = cols[0].selectbox("HER2 status", list(her2status.keys()))
    surgerytype = {"Other": 1, "Modified Radical Mastectomy": 2,
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4}
    Surgerytype = cols[1].selectbox("Surgery type", list(surgerytype.keys()))
    New_values = [Age,gender[Gender],protein_1,protein_2,protein_3,protein_4,tumorstage[Tumorstage],
                  histology[Histology],erstatus[ERstatus],prstatus[PRstatus],her2status[HER2status],
                  surgerytype[Surgerytype]]
    if st.button("Predict Result"):
        prediction = model.predict([New_values])
        if prediction == 1:
            st.success("Prediction : Alive")
        else:
            st.error("Prediction : Dead")




