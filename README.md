# iris_flower_dataset_ml_project
“This project leverages Principal Component Analysis (PCA) to build and evaluate a classification model on the Iris flower dataset. It focuses on using PCA for dimensionality reduction to predict Iris flower species based on their features. Includes data exploration, PCA-based model training, and performance evaluation.
## Installation

To get started with this project, you need to have Python installed on your system. You can then install the required packages using `pip`.

1. Clone the repository:

    ```bash
    git clone https://github.com/rajeev-gaur/iris_flower_dataset_ml_project.git
    cd iris-dataset-project
    ```

2. Create a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```
## Usage

To run the analysis and generate visualizations, execute the `iris_flower_dataset_ML.py` script and 'iris_visuallisation.py' scripts respectively.

```bash
python iris_flower_dataset_ML.py
python iris_visualisation.py 

The Iris dataset used in this project is included with the `sklearn` library. It consists of 150 samples of iris flowers, with 4 features for each sample:
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

The dataset includes three species of iris:
- Setosa
- Versicolor
- Virginica

 Results :The following results were obtained from the classification model:

- **Accuracy**: 0.91
- **Classification Report**:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        19
  versicolor       0.91      0.77      0.83        13
   virginica       0.80      0.92      0.86        13
    accuracy                           0.91        45
   macro avg       0.90      0.90      0.90        45
weighted avg       0.92      0.91      0.91        45
