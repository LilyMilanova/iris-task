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
- Python 3.10.16 
- TensorFlow (including tensorflow-macos and tensorflow-metal)
- PyTorch 
- Hugging Face Transformers and Datasets
- Scikit-learn and Scikit-multilearn 
- NLTK 
- Pandas 
- NumPy 
- Matplotlib 
- WordCloud 
- Optuna (for hyperparameter optimization)
- Accelerate
- Django (REST Framework)

## How to Use
1. Clone the repository.
2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```
    Note: Tested only on Macos with Apple M2 chip.

3. Create sample dataset for training:
    - The original file from ArXiv `arxiv-metadata-oai-snapshot.json` is pre-processed as JSON array containing only the `abstracts` and `categories` columns and saved in `arxiv-snapshot-training-json.json`
    - Then this file is used to extract samples of `<sample_size>%` percentage to be used for training
    - Three approaches were tested to extract samples. The current one takes equal number of entries from each category and adjusts if some categories are rare
   ```
   cd classifier/tools
   poetry run python create_sample.py
   ```
   Each dataset is stored in `/data/datasets` with the name ending in format `arxiv-snapshot-sample_<sample_size>%-json.json`

4. Train the model:
   - Specify `full_model_name` of base model in format `<repo_name>/<model_name>` e.g. `prajjwal1/bert-tiny`
   - Hyperparameters search can be invoked using `hyp_search` and `n_trials` flags of `MultiLabelClassificationModelTrainer`
   ```bash
   cd classifier/tools
   poetry run python train_model.py
   ```
   After training (best) model is saved in `/data/models/<model_name><sample_size>`. Sorted category to ID mapping are save for each trained model - only needed if sample size is small and not all categories are present. 

5. Run the API server:
   - choose a model from `/data/models` directory and configure it in `/classifier/article_classifier.json`.
    ```bash
    poetry run python manage.py runserver
    ```

6. Use the API to classify research article abstracts.

## Results
Evaluate the model using the validation set and analyze metrics for insights into performance.

## References
- [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [Hugging Face bert-tiny model](https://huggingface.co/prajjwal1/bert-tiny)
