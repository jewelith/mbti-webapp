import streamlit as st
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import joblib
from lemmatizer import Lemmatizer


personality_images = {
    "ISTJ": "istj_image.png",
    "ISFJ": "isfj_image.png",
    "INFJ": "infj_image.png",
    "INTJ": "intj_image.png",
    "ISTP": "istp_image.png",
    "ISFP": "isfp_image.png",
    "INFP": "infp_image.png",
    "INTP": "intp_image.png",
    "ESTP": "estp_image.png",
    "ESFP": "esfp_image.png",
    "ENFP": "enfp_image.png",
    "ENTP": "entp_image.png",
    "ESTJ": "estj_image.png",
    "ESFJ": "esfj_image.png",
    "ENFJ": "enfj_image.png",
    "ENTJ": "entj_image.png"
}



def clear_text(data):
    cleaned_text = []
    lemmatizer = WordNetLemmatizer()
    for sentence in data:
        sentence = sentence.lower()
        sentence = re.sub('https?://[^\s<>"]+|www\.[^\s<>"]+', ' ', sentence)
        sentence = re.sub('[^0-9a-z]', ' ', sentence)
        cleaned_text.append(sentence)
    return cleaned_text

def get_personality_description(personality_type):
    descriptions = {
        "ISTJ": "The Inspector - ISTJs are responsible organizers, driven to create and enforce order within systems and institutions.",
        "ISFJ": "The Protector - ISFJs are industrious caretakers, loyal to traditions and organizations.",
        "INFJ": "The Advocate - INFJs are empathetic visionaries, devoted to their values and committed to creating positive change.",
        "INTJ": "The Architect - INTJs are strategic thinkers, motivated to organize change and knowledge to bring about their vision.",
        "ISTP": "The Craftsman - ISTPs are observant mechanics, who enjoy working with their hands and analyzing how things work.",
        "ISFP": "The Composer - ISFPs are flexible and charming artists, who enjoy the present moment and new experiences.",
        "INFP": "The Healer - INFPs are imaginative idealists, guided by their own core values and beliefs.",
        "INTP": "The Thinker - INTPs are logical innovators, fascinated by the world of possibilities and driven by curiosity.",
        "ESTP": "The Dynamo - ESTPs are energetic thrill-seekers, who are adaptable, spontaneous, and resourceful.",
        "ESFP": "The Performer - ESFPs are vivacious entertainers, who charm and engage those around them with their enthusiasm.",
        "ENFP": "The Champion - ENFPs are enthusiastic advocates, who see life as full of possibilities and are driven by their values.",
        "ENTP": "The Visionary - ENTPs are enterprising explorers, who are innovative, resourceful, and see the world as full of possibilities.",
        "ESTJ": "The Supervisor - ESTJs are dependable organizers, who are driven to implement their plans and ensure that tasks are completed.",
        "ESFJ": "The Provider - ESFJs are conscientious helpers, who are motivated to help and provide for others, often putting their needs above their own.",
        "ENFJ": "The Teacher - ENFJs are charismatic leaders, who are highly empathetic and strive to create a positive impact on others.",
        "ENTJ": "The Commander - ENTJs are strategic leaders, who are assertive and enjoy taking charge to organize people and resources to achieve their goals."
    }
    return descriptions.get(personality_type, "Description not available.")


st.title('16 Personality Prediction')

st.write("The Myers-Briggs Type Indicator (MBTI) is a widely-used personality test that helps individuals understand their personality preferences based on four dichotomies: Extraversion (E) vs. Introversion (I), Sensing (S) vs. Intuition (N), Thinking (T) vs. Feeling (F), and Judging (J) vs. Perceiving (P). Each of the 16 personality types is a combination of these dichotomies. The MBTI is often used in various contexts, such as career counseling, team-building, and personal development. However, it's important to understand that the MBTI has been subject to criticism, particularly regarding its reliability and validity as a measure of personality. Some critics argue that it oversimplifies personality and lacks scientific rigor compared to other personality assessments. Nonetheless, many people still find value in the insights it provides into individual differences and preferences.")

user_input = st.text_area("Enter the last thing you posted:", "")

selected_model = st.selectbox("Select ML Model", ["Logistic Regression", "Multinomial Naive Bayes", "Linear SVC", "Random Forest", "Decesion Tree Classifier"])

if st.button('Predict'):
    if selected_model == "Logistic Regression":
        model = joblib.load('model_log.joblib')
    elif selected_model == "Multinomial Naive Bayes":
        model = joblib.load('model_multinomial_nb.joblib')
    elif selected_model == "Linear SVC":
        model = joblib.load('model_linear_svc.joblib')
    elif selected_model == "Random Forest":
        model = joblib.load('model_forest.joblib')
    elif selected_model == "Decesion Tree Classifier":
        model = joblib.load('model_tree.joblib')


    vectorizer = joblib.load('vectorizer.joblib')

    cleaned_user_input = clear_text([user_input])

    user_input_post = vectorizer.transform(cleaned_user_input).toarray()

    user_prediction = model.predict(user_input_post)

    target_encoder = joblib.load('target_encoder.joblib')


    predicted_user_type = target_encoder.inverse_transform(user_prediction)[0]


    personality_description = get_personality_description(predicted_user_type)

  
    st.write(f"Predicted Personality Type: {predicted_user_type}")
    st.write(f"Personality Description: {personality_description}")
   
    if predicted_user_type in personality_images:
        image_path = personality_images[predicted_user_type]

        col1, col2, col3 = st.columns([1, 1.85, 1])

        with col2:
            st.image(image_path, caption="Image corresponding to predicted personality type")

    else:
        st.write("Image not available for the predicted personality type.")

    import matplotlib.pyplot as plt
    import numpy as np

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(user_input_post)[0]
    else:
        probabilities = None
    if probabilities is not None:
        personality_types = list(target_encoder.classes_)
        y_pos = np.arange(len(personality_types))

        plt.figure(figsize=(10, 6))
        plt.barh(y_pos, probabilities, align='center')
        plt.yticks(y_pos, personality_types)
        plt.xlabel('Probability')
        plt.title('Probabilities of Personality Types')
        st.pyplot(plt)
    else:
        st.write("Probability information not available for the selected model.")

