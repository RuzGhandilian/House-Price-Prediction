# House Price Prediction using Neural Networks

This project focuses on predicting house prices using a neural network implemented with PyTorch. The dataset used is the **Housing Dataset**, which contains various features related to houses, such as area, number of bedrooms, and location. The goal is to build a model that accurately predicts house prices based on these features.

## Project Structure
- **Main File**: `House_prices.ipynb`
- **Data**: `../data/Housing.csv`
- **Model**: Saved as `../model/best_model.pth`

## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Analyzed the dataset to understand its structure, distribution, and correlations.
   - Visualized the distribution of house prices and identified outliers using boxplots.
   - Generated a correlation matrix to understand relationships between numeric features.

2. **Data Preprocessing**:
   - Handled outliers by capping extreme values in the `price` and `area` columns.
   - Encoded categorical variables using one-hot encoding.
   - Split the data into training and validation sets (80:20 ratio).
   - Normalized the data using `MinMaxScaler`.

3. **Model Architecture**:
   - A feedforward neural network with three hidden layers and dropout for regularization.
   - Hyperparameters such as learning rate and hidden layer sizes were tuned using a grid search approach.

4. **Training**:
   - Trained the model using the Adam optimizer and Mean Squared Error (MSE) loss.
   - Implemented early stopping and learning rate scheduling to prevent overfitting.
   - Evaluated the model using R² score and Root Mean Squared Error (RMSE).

5. **Results**:
   - Achieved a **Training RMSE** of **961,970.68** and **Training R²** of **0.6670**.
   - Achieved a **Validation RMSE** of **1,108,476.28** and **Validation R²** of **0.6959**.
   - Visualized the actual vs. predicted prices for both training and validation sets.

## Key Visualizations
- **Distribution of House Prices**: Histogram showing the distribution of the target variable.
- **Correlation Matrix**: Heatmap displaying correlations between numeric features.
- **Training and Validation Loss**: Line plot showing the loss over epochs.
- **Actual vs. Predicted Prices**: Scatter plots comparing actual and predicted prices for training and validation sets.

## Conclusion
The model demonstrates reasonable performance in predicting house prices, with a validation R² score of **0.6959**. The results indicate that the neural network can capture patterns in the data, but there is room for improvement, such as feature engineering or experimenting with more advanced architectures.

## How to Run
1. Ensure all dependencies are installed (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `torch`).
2. Open the `House_prices.ipynb` notebook and run the cells sequentially.
3. The trained model will be saved to `../model/best_model.pth`.
