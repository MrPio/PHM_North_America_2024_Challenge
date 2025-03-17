# PHM North America 2024 Challenge
Addressing [PHM North America 2024 Challenge](https://data.phmsociety.org/phm2024-conference-data-challenge/) with [Kolmogorov-Arnold Networks](https://github.com/KindXiaoming/pykan).


## 📙 Slideshows

<img height="150rem" align="right" src="https://github.com/user-attachments/assets/9d513014-9f84-4387-b68e-7c7941bad89d"/>

### 1️⃣ First review [`.PPTX`](https://github.com/user-attachments/files/19043133/Gruppo.A1.prima.revisione.pptx) [`.PDF`](https://github.com/user-attachments/files/19043132/Gruppo.A1.prima.revisione.pdf)
### 2️⃣ Second review [`.PPTX`](https://github.com/user-attachments/files/19074287/Gruppo.A1.seconda.revisione.pptx) [`.PDF`](https://github.com/user-attachments/files/19074286/Gruppo.A1.seconda.revisione.pdf)
### 3️⃣ Third review [`.PPTX`](https://github.com/user-attachments/files/19250163/Gruppo.A1.terza.revisione.pptx) [`.PDF`](https://github.com/user-attachments/files/19250162/Gruppo.A1.terza.revisione.pdf)

## 📙 How this repository is structured

This repository contains 14 Jupyter notebook, structured in according to the strategy devised to tackle the challenge, shown in the following map.

<p align="center">
    <img width="65%" src="img/strategy_map.png"/>
</p>

> [!IMPORTANT]  
> Before running the notebooks, make sure to extract the [`\dataset\0-original\dataset.zip`](/dataset/0-original/dataset.zip) archive and to run `pip install -r requirements.txt`. Then, you can begin running the notebooks in sequence, starting from [1.1](1-data_preprocessing/1.1-Exploratory_Data_Analysis.ipynb).

You can run the notebook through the jupyter lab running `jupyter lab` in the root directory, or use any IDE such as *PyCharm* or *Visual Studio Code* that handle that for you.

> [!NOTE]  
> As I hadn't included the `.csv` files due to their size, you should run all the notebooks one after the other. 

More specifically, the notebooks are:

- [`/1-data_preprocessing`](/1-data_preprocessing) - We perform some EDA in 1.1 to familiarise ourselves with the dataset, then proceed to generate derived features to improve subsequent training. We then use techniques common in statistics and machine learning, such as VIF, Pearson's correlation, ANOVA and Kruskal-Wallis tests, to Cherry-Pick these generated features.
   - [`1.1-Exploratory_Data_Analysis`](/1-data_preprocessing/1.1-Exploratory_Data_Analysis.ipynb) - This notebook shows the distribution of the 7 features in the samples, also taking into account their faulty state. A Principal Component Analysis is used to embed the 7 features into 2 and 3 directions in order to visualize the dataset in one plot. Here we find out that the dataset is made of 2 distinct clusters of samples, one for samples having $np \lt ng$ and the other for samples having $np \gt ng$. In conclusion, a skewness study is carried out to show that any 10% subsample does preserve the topology of the whole dataset.
   - [`1.2-Multicollinearity_Study_With_VIF`](/1-data_preprocessing/1.2-Multicollinearity_Study_With_VIF.ipynb) - We prove in this notebook that there exists a multicollinearity problem in the original dataset. More specifically, we find out that $np\approx  -1.06\times ng + 165$.
   - [`1.3-Feature_Extraction`](/1-data_preprocessing/1.3-Feature_Extraction.ipynb) - Here polynomial features up to degree 3, np/ng ratio, density altitude and air density normalization are generated. In total, we end up with 81 features. A selection is needed.
   - [`1.4-Correlation_Pruning`](/1-data_preprocessing/1.4-Correlation_Pruning.ipynb) - Therefore, using the Pearson we begin with pruning all highly correlated features. Then, we explain why some of them present a correlation of almost 1.
   - [`1.5-Cherry_Picking_For_Regression`](/1-data_preprocessing/1.5-Cherry_Picking_For_Regression.ipynb) - We combine Pearson's correlation with $torque_{target}$ and ANOVA F-test to select the 3 most discriminant features.
   - [`1.6-Cherry_Picking_For_Classification`](/1-data_preprocessing/1.6-Cherry_Picking_For_Classification.ipynb) - We combine Pearson's correlation with $faulty$, ANOVA F-test and Kruskal-Wallis test to select the 5 most discriminant features.
- [`/2-torque_target_probabilistic_regression`](/2-torque_target_probabilistic_regression)
   - [`2.1-Synthetic_Dataset`](/2-torque_target_probabilistic_regression/2.1-Synthetic_Dataset.ipynb)
   - [`2.2-Torque_Target_Regression_Stochastic`](/2-torque_target_probabilistic_regression/2.2-Torque_Target_Regression_Stochastic.ipynb)
   - [`2.3-Using_Learned_Regressor_To_Predict_Torque_Target`](/2-torque_target_probabilistic_regression/2.3-Using_Learned_Regressor_To_Predict_Torque_Target.ipynb)
- [`/3-fault_detection`](/3-fault_detection)
   - [`3.1-Fault_Detection`](/3-fault_detection/3.1-Fault_Detection.ipynb)
- [`/4-qualitative_evaluation`](/4-qualitative_evaluation)
   - [`4.1-Qualitative_Evaluation_PCA`](/4-qualitative_evaluation/4.1-Qualitative_Evaluation_PCA.ipynb)
   - [`4.2-KNN_In_PCA_Domain`](/4-qualitative_evaluation/4.2-KNN_In_PCA_Domain.ipynb)
   - [`4.3-Qualitative_Evaluation_t-SNE`](/4-qualitative_evaluation/4.3-Qualitative_Evaluation_t-SNE.ipynb)
   - [`4.4-Qualitative_Evaluation_experiments`](/4-qualitative_evaluation/4.4-Qualitative_Evaluation_experiments.ipynb)