import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import joblib
import concurrent.futures
import numpy as np

# Load data
x = np.load(r"x_all.npy")
y = np.load(r"y_all.npy")

def train_model(j, i, model_class, model_params, model_dir):
    """
    Train and save a single model instance.

    Parameters:
    - j: int, iteration index for model naming and random state.
    - i: int, outer loop index (could represent a different split or seed).
    - model_class: class, the classifier class from scikit-learn.
    - model_params: dict, parameters to initialize the classifier.
    - model_dir: str, directory path to save the trained model.
    """
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=i
    )
    
    # Resample the training data
    x_resampled, y_resampled = resample(
        x_train, y_train, replace=True, random_state=j
    )
    
    # Balance the sample using SMOTE
    smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=j)
    x_balanced, y_balanced = smote.fit_resample(x_resampled, y_resampled)
    
    # Initialize a new model instance
    model = model_class(**model_params)
    
    # Train the model
    model.fit(x_balanced, y_balanced)
    
    # Save the trained model
    model_path = os.path.join(model_dir, f'model_{j+1}.joblib')
    joblib.dump(model, model_path)

def main():
    baggings = {
    # 'MLP': {
    #     'class': MLPClassifier,
    #     'params': {'hidden_layer_sizes': (100,), 'max_iter': 1000}
    # },
    # 'DecisionTree': {
    #     'class': DecisionTreeClassifier,
    #     'params': {'criterion': 'gini'}
    # },
    # 'Knn': {
    #     'class': KNeighborsClassifier,
    #     'params': {'n_neighbors': 7, 'n_jobs': -1}
    # },
    'NaiveBayes': {
        'class': GaussianNB,
        'params': {}
    }
}
    
    num_outer_iterations = 30
    num_inner_iterations = 100  # Number of models per outer iteration
    
    for model_name, model_info in baggings.items():
        model_class = model_info['class']
        model_params = model_info['params']
        
        for i in range(num_outer_iterations):
            model_dir = os.path.join("Balanced\Bagging2", model_name, str(i))
            os.makedirs(model_dir, exist_ok=True)  # Create the directory if it doesn't exist
            print(f"Training models in directory: {model_dir}")
            
            # Prepare arguments for parallel execution
            args = [
                (j, i, model_class, model_params, model_dir)
                for j in range(num_inner_iterations)
            ]
            
            # Use ProcessPoolExecutor for parallelism
            with concurrent.futures.ProcessPoolExecutor() as executor:
                # Map the function to the arguments
                # Using executor.submit in a loop for better control and error handling
                futures = [
                    executor.submit(train_model, *arg)
                    for arg in args
                ]
                
                # Optionally, handle results or exceptions
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"Generated an exception: {exc}")

if __name__ == "__main__":
    main()
