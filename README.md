# Prerequisites:
**Python _3.10.2_**
> pip3 install -r requirements.txt
### Please evaluate the code on the second method (main.py file and scripts in the src directory instead of the notebooks)
# How to run
## Method 1) Jupyter notebooks
### 1) Install Jupyter notebook
> pip3 install notebook
### 2) Update the path to fea.csv in the notebooks (for visualiztion notebook update the path for both csv files as well)
### 3) Run the notebooks (You can tell by name which notebook is which)  

## Method 2) main.py file
### 1) Simply update the path to fea.csv in the file and run the code.
### 2) By default the main.py file computes the KNN after performing PCA. To change that behaviour you can set the value of the variable in the main.py file:
```pca = True``` to ```pca = False```
### 3) You can also change whether you want to visualize the covariance matrices or not. If you have set pca to False as above then you won't see the matrices. But if you want to do PCA without plotting the matrices, set the variable below the pca variable to False.
```pca_visualize = True``` to ```pca_visualize = False```



# Difference between both
1) The notebooks save the resulting dataframes as csv files which are then used to visualize the results with the **_visualize_results_** notebook. The main.py file skips the saving step and straight away visualizes the results.

2) The notebooks are not that optimized and there may be redundant code but they are easy to see the flow of the code. The main.py file and the source code files in the **src** directory are written to remove redundancy as best as possible, with **PEP8** guidelines and in a more **modular** fashion. They are easy to **test** as well and may require only a few modification when the code is being productionized.