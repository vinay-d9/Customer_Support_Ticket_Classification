"""Streamlit interface for customer support ticket classification."""

from pathlib import Path

import joblib
import streamlit as st

from train_model import preprocess_text


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "ticket_classifier.pkl"
VECTORIZER_FILE = BASE_DIR / "tfidf_vectorizer.pkl"
CONFIDENCE_THRESHOLD = 0.60
DEPARTMENT_BY_CATEGORY = {
    "Billing": "Billing Department",
    "Technical Support": "Technical Support Team",
    "Account Access": "Account Support Team",
    "Order Issues": "Order Management Team",
}


@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            "Model files are missing. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_FILE), joblib.load(VECTORIZER_FILE)


st.set_page_config(page_title="Customer Support Ticket Classification")
st.title("Customer Support Ticket Classification")
st.write(
    "Enter a support request to classify it and demonstrate confidence-based routing."
)
st.caption(
    "Confidence is an estimate, not a guarantee. Unfamiliar requests may require manual review."
)

ticket_text = st.text_area(
    "Customer support request",
    placeholder="Example: I was charged twice for my subscription this month.",
    height=140,
)

if st.button("Classify ticket", type="primary"):
    if not ticket_text.strip():
        st.warning("Please enter a customer support request before classifying.")
    else:
        try:
            model, vectorizer = load_artifacts()
            features = vectorizer.transform([preprocess_text(ticket_text)])
            category = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]
            confidence = float(max(probabilities))

            st.subheader("Classification result")
            st.write(f"**Predicted category:** {category}")
            st.write(f"**Confidence:** {confidence:.1%}")

            if confidence >= CONFIDENCE_THRESHOLD:
                department = DEPARTMENT_BY_CATEGORY.get(category)
                if department is None:
                    st.error("The predicted category has no configured department.")
                else:
                    st.success(f"**Assigned department:** {department}")
            else:
                st.warning(
                    "**Assigned department:** Manual Review Required. "
                    "This ticket will not be routed automatically."
                )
        except (FileNotFoundError, ValueError, OSError) as error:
            st.error(str(error))
