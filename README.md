# Cancer_Detection
Classification Problem using Neural Network

## Dataset

A) About Dataset:

1. Dataset is from Kaggle. It's Breat Cancer Wisconsin ( Diagnostic ).
2. Dataset has total 32 features. 
3. Ten real-valued features are computed or each cell : radius (mean of distances from center to points on the perimeter), texture (standard deviation of gray-scale values), perimeter, area, smoothness (local variation in radius lengths), compactness (perimeter^2 / area - 1.0), concavity (severity of concave portions of the contour), concave points (number of concave portions of the contour), symmetry, fractal dimension ("coastline approximation" - 1).
4. We will map Diagnosis (M = malignant, B = benign) feature in 1 for malignant and 0 for benign for easier calculation.

   You can find more imformation about the dataset here:
   https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

B) Cleaning Dataset:

1. Deleted dummy labels “ Unnamed: 32 ” and “id”.
2. id information did not have any use and unnamed label had no data.

C) Feature Selection:

   To find out which two features had most effect on classification, Recursive Feature Elimination (RFE.py) is used.

   The Recursive Feature Elimination (or RFE) works by recursively removing attributes and building a model on those     attributes that remain. It uses the model accuracy to identify which attributes (and combination of attributes) contribute the most to predicting the target attribute.

## Neural Network

1. Read data.csv file using pandas library in python.
2. Input layer and Output layer dimentionality is two. Best two features has been used as input nodes.
3. Using 3 hidden layer
4. Using Sigmoid function as activation funtion
5. It's a two classification problem. 
6. Neural Network will classify the problem in either M (malignant) or B (benign).
7. Diagnosis feature is used as label.


