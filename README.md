# Student Loan Risk Prediction with Deep Learning

## Project Overview
This project applies deep learning techniques to predict student loan repayment risk based on various student and loan attributes. 
Using TensorFlow and Keras, the model helps identify factors that contribute to successful loan repayment, which can be valuable for educational financial institutions to assess loan applications.

## Dataset Information
The dataset (`student-loans.csv`) contains information about student borrowers with the following features:
- `payment_history` - Historical data on previous payment patterns
- `location_parameter` - Geographic factor affecting employment prospects
- `stem_degree_score` - Indicator of STEM education (higher values for STEM degrees)
- `gpa_ranking` - Academic performance metric
- `alumni_success` - Metric of past graduates' success
- `study_major_code` - Field of study identifier
- `time_to_completion` - Time taken to complete education (in months)
- `finance_workshop_score` - Performance in financial literacy training
- `cohort_ranking` - Relative standing within graduation cohort
- `total_loan_score` - Debt burden metric
- `financial_aid_score` - Non-loan financial assistance metric
- `credit_ranking` - Target variable (1 = good credit risk, 0 = poor credit risk)

## Technical Implementation

### Environment Requirements
- Python 3.8+
- TensorFlow 2.x
- Pandas
- Scikit-learn
- NumPy
- Matplotlib

### Project Structure
```
.
├── student_loans_with_deep_learning.ipynb   # Main Jupyter notebook
├── student_loans.keras   				                      # Saved Keras model
├── student-loans.csv                      					  # Dataset 
└── README.md                                				  # This file
```

### Model Architecture
The neural network model consists of:
- Input layer: 11 features
- First hidden layer: 6 neurons with ReLU activation
- Second hidden layer: 3 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation for binary classification

### Data Preprocessing
1. Features and target variable separation
2. Training and testing data split (75%/25%)
3. Feature standardization using StandardScaler

### Model Performance
- Loss (Binary Cross-Entropy): ~0.50
- Accuracy: ~73.5%
- Precision, Recall, and F1-scores are balanced for both classes

## Usage Instructions

### Running the Notebook
1. Clone this repository
2. Install required dependencies
3. Open and run `student_loans_with_deep_learning.ipynb` in Jupyter, Google Colab, or similar

### Using the Trained Model
```python
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model
model = tf.keras.models.load_model('student_loans.keras')

# Prepare your data (must have the same features as training data)
# Example:
new_data = pd.DataFrame({
    'payment_history': [8.2],
    'location_parameter': [0.65],
    'stem_degree_score': [0.32],
    # ... other features
})

# Scale the data using the same scaler parameters as during training
# Note: In production, you would need to save and load the scaler
scaler = StandardScaler()
# Either fit on training data or load saved scaler parameters
new_data_scaled = scaler.transform(new_data)

# Make predictions
predictions = model.predict(new_data_scaled)
risk_score = predictions[0][0]
risk_category = 1 if risk_score > 0.5 else 0
print(f"Risk Score: {risk_score:.2f}, Category: {'Good Risk' if risk_category == 1 else 'Poor Risk'}")
```

## Future Enhancements
- Hyperparameter tuning to improve model performance
- Feature importance analysis to identify key risk factors
- Exploration of more complex architectures or ensemble methods
- Integration with a recommendation system for student loans
- Deployment as a web service for real-time risk assessment

## Ethical Considerations
This model should be used as a supplementary tool for risk assessment, not as the sole decision-maker for student loan approvals. Care should be taken to:
- Regularly test for bias in predictions across different demographic groups
- Ensure transparency in the decision-making process
- Provide explanations for risk assessments to applicants
- Balance model predictions with human judgment and context

## License
See License file (Mozilla Public License Version 2.0) 

## Contributors
For any questions or requests for use contact RStover @ Gmail (dot) com 