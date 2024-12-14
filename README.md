# Research Article Classification

## Objective
Classify research articles into predefined categories (e.g., biology, chemistry, physics, computer science, and social sciences) based on their content.

## Dataset
The project uses the public [arXiv dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv). Abstracts are utilized for classification, and subject categories are used for class association.

## Features
- **Exploratory Data Analysis (EDA):** Gain insights into the dataset structure and characteristics.
- **Data Preprocessing:**
  - Cleaning and normalization of text (removing special characters, converting to lowercase, etc.).
  - Removing stop words and applying stemming/lemmatization.
  - Text vectorization for machine learning readiness.
- **Model Training:**
  - Trains an NLP text classification model using TensorFlow or Keras with pre-trained models from [Hugging Face](https://huggingface.co/models?library=tf&language=en&license=license:apache-2.0&sort=downloads&search=bert-base).
- **Model Evaluation:**
  - Performance metrics: Accuracy, Precision, Recall, and F1-Score.
- **Hyperparameter Tuning:** Optimize the model for better performance.
- **API Implementation:**
  - Built with Django REST Framework.
  - Allows input of research article abstracts and returns predicted categories.

## Technologies Used
- Python
- TensorFlow/Keras
- Scikit-learn
- NLTK
- Django REST Framework
- Poetry

## How to Use
1. Clone the repository.
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
3. Train the model:
   ```bash
   poetry run python train_model.py
   ```
4. Run the API server:
   ```bash
   poetry run python manage.py runserver
   ```
5. Use the API to classify research article abstracts.

## Results
Evaluate the model using the validation set and analyze metrics for insights into performance.

## References
- [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [Hugging Face bert-tiny model](https://huggingface.co/prajjwal1/bert-tiny)
