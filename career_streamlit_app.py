import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from llama_cpp import Llama
import PyPDF2
import spacy

# --- Page setup ---
st.set_page_config(page_title="Career Genie", page_icon="🚀", layout="wide")

# --- Global colorful styling ---
st.markdown("""
    <style>
        /* Gradient background for the whole app */
        .stApp {
            background: linear-gradient(135deg, #ffdde1 0%, #ee9ca7 25%, #a1c4fd 50%, #c2e9fb 75%, #d4fc79 100%);
            color: #000000;
            font-family: 'Poppins', sans-serif;
        }

        /* Tabs styling */
        div[data-baseweb="tab-list"] button {
            background-color: #ffffff33 !important;
            color: #003366 !important;
            border-radius: 10px;
            border: 2px solid #fff;
            margin-right: 5px;
            font-weight: bold;
        }
        div[data-baseweb="tab-list"] button[aria-selected="true"] {
            background: linear-gradient(90deg, #ff758c, #ff7eb3);
            color: white !important;
            box-shadow: 0px 0px 10px #ffb6c1;
        }

        /* Headings */
        h1, h2, h3 {
            color: #2b2d42;
            text-shadow: 1px 1px 3px #ffffffaa;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg, #43cea2, #185a9d);
            color: white;
            border-radius: 12px;
            height: 3em;
            width: 100%;
            font-size: 1em;
            font-weight: 600;
            box-shadow: 2px 2px 8px #00000033;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #11998e, #38ef7d);
            transform: scale(1.02);
        }

        /* Input boxes */
        .stSelectbox, .stNumberInput {
            border-radius: 10px !important;
        }

        /* Info & success boxes */
        .stSuccess, .stInfo {
            border-radius: 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load ML Model ---
@st.cache_resource
def load_model():
    data = {
        'Interest': ['AI', 'Design', 'Biology', 'Business', 'Programming', 'Data', 'AI', 'Design', 'Finance', 'Programming'],
        'Skill': ['Python', 'Creativity', 'Research', 'Management', 'Java', 'Statistics', 'ML', 'Photoshop', 'Excel', 'C++'],
        'Academic_Score': [85, 78, 90, 88, 82, 87, 92, 75, 80, 84],
        'Career': ['Data Scientist', 'Graphic Designer', 'Biotechnologist', 'Business Analyst',
                   'Software Engineer', 'Data Analyst', 'AI Engineer', 'UI/UX Designer',
                   'Financial Analyst', 'Software Developer']
    }
    df = pd.DataFrame(data)
    label_encoders = {}
    for col in ['Interest', 'Skill', 'Career']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df[['Interest', 'Skill', 'Academic_Score']]
    y = df['Career']
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, label_encoders, df

model, label_encoders, df = load_model()

# --- Load LLaMA Model ---
@st.cache_resource
def load_llama():
    return Llama(model_path="C:/Users/Arvind/Downloads/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

llm = load_llama()

# --- NLP for CV Analyzer ---
nlp = spacy.load("en_core_web_sm")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🌈 ML Career Genie", "🤖 Free AI Chat Assistant", "📜 Smart CV Analyzer"])

# --- Tab 1: Career Recommender ---
with tab1:
    st.markdown("<div style='background-color:#ffe9e3;padding:25px;border-radius:15px;'>", unsafe_allow_html=True)
    st.title("🎓 Find Your Perfect Career Path")
    interest = st.selectbox("Select Your Interest", label_encoders['Interest'].classes_, key="interest")
    skill = st.selectbox("Select Your Key Skill", label_encoders['Skill'].classes_, key="skill")
    score = st.number_input("Enter Academic Score", min_value=0, max_value=100, value=85, key="score")

    if st.button("💼 Recommend My Career", key="predict_btn"):
        sample = pd.DataFrame({"Interest": [interest], "Skill": [skill], "Academic_Score": [score]})
        for col in ['Interest', 'Skill']:
            sample[col] = label_encoders[col].transform(sample[col])
        pred = model.predict(sample)
        career_name = label_encoders['Career'].inverse_transform(pred)[0]
        st.success(f"🌟 *Your Ideal Career:* {career_name}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 2: AI Chat Assistant ---
with tab2:
    st.markdown("<div style='background-color:#e0f7fa;padding:25px;border-radius:15px;'>", unsafe_allow_html=True)
    st.title("💬 Ask Anything — Career Chatbot")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! Ask me anything about your career."}]
    user_input = st.text_input("Type your question here 👇")
    if st.button("Send", key="chat_btn") and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            full_prompt = f"You are a helpful AI career mentor.\nUser: {user_input}\nAI:"
            result = llm(full_prompt, max_tokens=256, temperature=0.7)
            answer = result["choices"][0]["text"].strip()
        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for msg in st.session_state.chat_history:
        if msg["role"] == "assistant":
            st.markdown(f"💡 **Bot:** {msg['content']}")
        else:
            st.markdown(f"🙋 **You:** {msg['content']}")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Tab 3: CV Analyzer ---
with tab3:
    st.markdown("<div style='background-color:#fce4ec;padding:25px;border-radius:15px;'>", unsafe_allow_html=True)
    st.title("📄 CV Skill & Career Analyzer")
    uploaded_file = st.file_uploader("Upload your Resume (PDF only)", type=["pdf"])
    if uploaded_file is not None:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        all_text = "".join([page.extract_text() for page in pdf_reader.pages])

        st.subheader("🧾 Extracted Text Preview:")
        st.write(all_text[:1500] + ("..." if len(all_text) > 1500 else ""))

        doc = nlp(all_text)
        skills = set([ent.text for ent in doc.ents if ent.label_ in {"ORG","NORP","PERSON","GPE","SKILL"}])
        st.success("✅ Detected Keywords / Entities:")
        st.write(", ".join(skills) if skills else "No clear skills found. Try a more detailed CV.")

        if st.button("🔍 Get Career Advice for My CV"):
            prompt = (
                f"Here is my resume:\n{all_text[:800]}\n"
                "Based on this, suggest the top 3 careers I might fit best for."
            )
            with st.spinner("Analyzing your CV with Mistral..."):
                result = llm(prompt, max_tokens=256, temperature=0.7)
                answer = result["choices"][0]["text"].strip()
            st.info(f"🎯 **AI Career Suggestion:**\n\n{answer}")
    st.markdown("</div>", unsafe_allow_html=True)




