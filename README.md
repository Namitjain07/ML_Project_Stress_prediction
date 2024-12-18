# Stress Prediction

This repository contains a machine learning project aimed at building and evaluating predictive models using various algorithms. The project leverages Python libraries such as Pandas, Scikit-learn, Matplotlib, and XGBoost.

## Project Overview
The goal of this project is to preprocess data, train multiple machine learning models, and evaluate their performance using various metrics. The project includes the following components:

- Data preprocessing and feature engineering
- Training multiple machine learning models
- Evaluating models using performance metrics
- Saving and loading trained models for deployment

## Features
- **Preprocessing**: Handles scaling, encoding, and splitting of datasets.
- **Models**:
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - Support Vector Machines (SVM)
  - Naive Bayes
  - XGBoost
  - AdaBoost
  - Perceptron
- **Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Classification Reports
  - Confusion Matrices

## Installation

1. Clone the repository:
   ```bash
   git clone "https://github.com/satwikgarg2022461/ML_Project_Stress_prediction.git"
   cd your-repository
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your dataset in the appropriate directory (e.g., `data/`).

2. Modify the notebook or script to point to your dataset path.

3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

4. Follow the steps in the notebook to preprocess data, train models, and evaluate results.

## File Structure
- `main.ipynb`: The main notebook containing the project code.
- `data/`: Directory for input datasets.
- `models/`: Directory for saving trained models.
- `outputs/`: Directory for storing evaluation results.
- `requirements.txt`: File specifying project dependencies.

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- pickle

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Contributing
Feel free to submit issues or pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- Python community for their amazing tools and libraries.

