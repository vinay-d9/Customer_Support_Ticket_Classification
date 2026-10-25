# Customer Support Ticket Classification

This college-level NLP project classifies customer support requests into **Billing**, **Technical Support**, **Account Access**, or **Order Issues**, then demonstrates confidence-based department routing.

## Workflow

`customer_support_tickets.csv` contains 480 labeled examples with `ticket_text` and `category` columns. `train_model.py`:

1. Inspects shape, columns, samples, missing values, duplicates, and category counts.
2. Normalizes whitespace and case, removes invalid rows, exact duplicates, and highly similar messages.
3. Creates an 80:20 stratified train-test split with a fixed random seed.
4. Fits a TF-IDF vectorizer on training text only.
5. Trains Logistic Regression and Multinomial Naive Bayes.
6. Reports macro precision, macro recall, macro-F1, classification reports, and labeled confusion matrices.
7. Selects the highest macro-F1 model and saves it with its fitted vectorizer.

The Streamlit app loads `ticket_classifier.pkl` and `tfidf_vectorizer.pkl` without retraining. It shows the predicted category and class-probability confidence. Predictions below the configurable 60% threshold are sent to manual review; otherwise the category is mapped to a department:

| Category | Department |
| --- | --- |
| Billing | Billing Department |
| Technical Support | Technical Support Team |
| Account Access | Account Support Team |
| Order Issues | Order Management Team |

## Run the project

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Run `python train_model.py` successfully before launching Streamlit. Training creates `ticket_classifier.pkl`, `tfidf_vectorizer.pkl`, and the two confusion-matrix PNG files.
