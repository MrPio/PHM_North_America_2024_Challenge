# PHM North America 2024 Challenge
<a href="https://data.phmsociety.org/phm-2024-data-challenge-submission-area/"><img src="https://img.shields.io/badge/PHM-Scoreboard-blue?style=for-the-badge" /></a>

Addressing [PHM North America 2024 Challenge](https://data.phmsociety.org/phm2024-conference-data-challenge/) with [Kolmogorov-Arnold Networks](https://github.com/KindXiaoming/pykan).

### 🏆 [J3C IFAC] Won ICONS Best Paper Award!

#### 📗 Paper (J3C IFAC - ICONS) - *Using Kolmogorov-Arnold Networks for an Interpretable and Continual Fault Detection of Helicopter Turbine Engine* [`.PDF`](slideshows/Using_Kolmogorov_Arnold_Networks_for_an_Interpretable_and_Continual_Fault_Detection_of_Helicopter_Turbine_Engines.pdf) 
#### 📗 Conference Slideshow (J3C IFAC - ICONS) - *Addressing PHM North America 2024 Challenge with Kolmogorov-Arnold Networks* [`.PPTX`](slideshows/Addressing PHM North America 2024 Challenge with Kolmogorov-Arnold Networks.pptx) 
#### 📘 Thesis - *Addressing PHM North America 2024 Challenge with Kolmogorov-Arnold Networks* [`.PDF`](slideshows/Addressing-PHM-North-America-2024-Challenge-With-Kolmogorov-Arnold-Networks.pdf) 
#### 📙 Review 1 - *Preliminary SOTA study* [`.PPTX`](https://github.com/user-attachments/files/19043133/Gruppo.A1.prima.revisione.pptx) [`.PDF`](slideshows/1-Preliminary_SOTA_study.pdf) 
#### 📙 Review 2 - *EDA and experiments with KANs* [`.PPTX`](https://github.com/user-attachments/files/19074287/Gruppo.A1.seconda.revisione.pptx) [`.PDF`](slideshows/2-EDA-And-Experiments-With-KANs.pdf) 
#### 📙 Review 3 - *Putting it all together* [`.PPTX`](https://github.com/user-attachments/files/19250163/Gruppo.A1.terza.revisione.pptx) [`.PDF`](slideshows/3-Putting-It-All-Together.pdf) 

## 📦 TL;DR
The goal of this project is to introduce Kolmogorov-Arnold Networks in the field of fault detection. There are two main advantages:
- *Interpretability*: as shown in the figures below, **KAN is far more interpretable than the MLP**. This comes at the cost of smaller accuracy.
- *Continual learning*: as proved in a [previous study of mine](https://github.com/MrPio/KAN-Continual_Learning_tests), **KANs suffer less from catastrophic forgetting than MLPs**.

### KAN
<div align="center">
  <img width="40%" align="center" src="https://github.com/user-attachments/assets/8821ca12-8198-4cfc-95b0-4e35e3b0a790"/>
  <img width="40%" align="center" src="https://github.com/user-attachments/assets/cce4d144-8994-4438-92b4-f5cfe005deb1"/>
</div>
<!-- <p align="center"><img height="275rem" align="center" src="https://github.com/user-attachments/assets/976846cd-c8ea-4fb1-a0da-67574bfb8b88"/></p> -->

### MLP
<p align="center"><img height="325rem" align="center" src="https://github.com/user-attachments/assets/1195ec61-f18a-4f9b-8018-719875e902e7"/></p>
<p align="center"><img height="325rem" align="center" src="https://github.com/user-attachments/assets/15c5e78f-4ac6-4d11-b59b-25e617b3c012"/></p>

### Domain-IL protocol
Non-trivial KANs outperform MLPs with the same trainable parameters in resilience to catastrophic forgetting.
<p align="center"><img height="325rem" align="center" src="https://github.com/user-attachments/assets/d3b4954d-6a7d-4049-b319-e4508ba22559"/></p>

## ⚙️ How to run

<img height="150rem" align="right" src="https://github.com/user-attachments/assets/9d513014-9f84-4387-b68e-7c7941bad89d"/>

### Prerequisites

- **Python 3.7+**
- **pip**
- Optionally, [**virtualenv**](https://virtualenv.pypa.io/) for managing environments

> [!IMPORTANT]  
> Before running the notebooks, make sure to extract the [`\dataset\0-original\dataset.zip`](/dataset/0-original/dataset.zip) archive!

1. [Optional] You can create and enter a virtual env with
  ```shell
  python3 -m venv venv
  source venv/bin/activate # "venv\Scripts\activate" on Windows
  ```
2. Run `pip install -r requirements.txt` in the root of the repository.
3. Run `jupyter lab` in the root of the repository to enter the web-based jupyter environment or use the tools provided by your favourite IDE.

You can now begin running the notebooks in sequence, starting from [1.1](1-data_preprocessing/1.1-Exploratory_Data_Analysis.ipynb).

> [!TIP]  
> To give you an idea of how much time each cell takes to complete, I have given the execution time on my *i7-10750H* as a comment above each cell. Please also note that **no GPU is required to run these notebooks**, even the training should work fine on the CPU, as the networks involved are very small.

## 📌 How this repository is structured

This repository contains 14 Jupyter notebook, structured in according to the strategy devised to tackle the challenge, shown in the following map.

<p align="center">
    <img width="80%" src="https://github.com/user-attachments/assets/37402303-b016-4cd3-80e5-1c7d0b869757"/>
</p>


You can run the notebook through the jupyter lab running `jupyter lab` in the root directory, or use any IDE such as *PyCharm* or *Visual Studio Code* that handle that for you.

> [!NOTE]  
> As I haven't included the `.csv` files due to their size, you should run all the notebooks one after the other, in order.

More specifically, the notebooks are:

- [`/1-data_preprocessing`](/1-data_preprocessing) - We perform some EDA in 1.1 to familiarize ourselves with the dataset, then proceed to generate derived features to improve subsequent trainings. We then use techniques common in statistics and machine learning, such as VIF, Pearson's correlation, ANOVA and Kruskal-Wallis tests, to Cherry-Pick these generated features.
   - [`1.1-Exploratory_Data_Analysis`](/1-data_preprocessing/1.1-Exploratory_Data_Analysis.ipynb) - This notebook shows the distribution of the 7 features in the samples, also taking into account their faulty state. A Principal Component Analysis is used to embed the 7 features into 2 and 3 directions in order to visualize the dataset in one plot. Here we find out that the dataset is made of 2 distinct clusters of samples, one for samples having $np \lt ng$ and the other for samples having $np \gt ng$. In conclusion, a skewness study is carried out to show that any 10% subsample does preserve the topology of the whole dataset.
   - [`1.2-Multicollinearity_Study_With_VIF`](/1-data_preprocessing/1.2-Multicollinearity_Study_With_VIF.ipynb) - We prove in this notebook that there exists a multicollinearity problem in the original dataset. More specifically, we find out that $np\approx  -1.06\times ng + 165$.
   - [`1.3-Feature_Extraction`](/1-data_preprocessing/1.3-Feature_Extraction.ipynb) - Here polynomial features up to degree 3, np/ng ratio, density altitude and air density normalization are generated. In total, we end up with 81 features. A selection is needed.
   - [`1.4-Correlation_Pruning`](/1-data_preprocessing/1.4-Correlation_Pruning.ipynb) - Therefore, using the Pearson we begin with pruning all highly correlated features. Then, we explain why some of them present a correlation of almost 1.
   - [`1.5-Cherry_Picking_For_Regression`](/1-data_preprocessing/1.5-Cherry_Picking_For_Regression.ipynb) - We combine Pearson's correlation with $torque_{target}$ and ANOVA F-test to select the 3 most discriminant features.
   - [`1.6-Cherry_Picking_For_Classification`](/1-data_preprocessing/1.6-Cherry_Picking_For_Classification.ipynb) - We combine Pearson's correlation with $faulty$, ANOVA F-test and Kruskal-Wallis test to select the 5 most discriminant features.
- [`/2-torque_target_probabilistic_regression`](/2-torque_target_probabilistic_regression) - In this section we solve the probabilistic regression problem. For each sample of the original data set, we want the network to be able to predict the PDF of the $trq_{target}$.
   - [`2.1-Preliminary_Study_On_Synthetic_Dataset`](/2-torque_target_probabilistic_regression/2.1-Preliminary_Study_On_Synthetic_Dataset.ipynb) - This notebook has the purpose of generating a sample data set for the training of a stochastic network. Can a PyKAN network really solve a probabilistic regression task?
   - [`2.2-Torque_Target_Regression`](/2-torque_target_probabilistic_regression/2.2-Torque_Target_Regression.ipynb) - In this notebook we're going to train a `PyKAN`, an `EfficientKAN`, and a `MLP` to learn a function that maps each set of input features to its own probability distribution. We will assume that the underlying process has many small, independent disturbances. Thus, Leveraging the Central limit theorem, we can say that the error between the model's prediction and **the actual observed value is likely to be normal**.
   - [`2.3-Generating_Torque_Margin_Predictions`](/2-torque_target_probabilistic_regression/2.3-Generating_Torque_Margin_Predictions.ipynb) - The PyKAN trained in 2.2 is here loaded and used to generate predictions over the entire dataset.
- [`/3-fault_detection`](/3-fault_detection) - In this section we train a PyKAN and a MultiKAN, i.e. a KAN with a multiplication node, to solve the fault detection task. In addition to the features selected in 1.6, the network is also fed with the $trq_{margin}$ predictions generated in 2.3.
   - [`3.1-Fault_Detection`](/3-fault_detection/3.1-Fault_Detection.ipynb) - Now that we've trained a network to predict the $trq_{margin}$, we will move on onto the fault detection task. In this notebook we'll train a `PyKAN` network to distinguish between *faulty* and *not-faulty* operational regimes.
   - [`3.2-Generating_Faulty_Predictions`](/3-fault_detection/3.2-Generating_Faulty_Predictions.ipynb) - The PyKAN trained in 3.1 is here loaded and used to generate predictions over the entire dataset.
   - [`3.3-Explaining_KAN_With_Shapley_Value`](/3-fault_detection/3.3-Explaining_KAN_With_Shapley_Value.ipynb) - The PyKAN trained in 3.1 is here loaded and explaining through the Shapley value technique. Originating from game theory, the Shapley value is a powerful tool in the field of eXplainableAI. Through an iterative approximation, the features of the samples are randomly permuted and the distance in their prediction is used to understand how the model weights each of them to assert the engine's faulty state.
- [`/4-qualitative_evaluation`](/4-qualitative_evaluation) - Does the trained PyKAN actually generalize over the test set? Unfortunately, we don't have the ground truth for this yet, so a qualitative analysis is needed.
   - [`4.1-Data_Drift_And_Outliers_Analysis`](/4-qualitative_evaluation/4.1-Data_Drift_And_Outliers_Analysis.ipynb) - Data drift seriously affects the test set. How is the KAN prediction distributed across the unseen feature regions?
   - [`4.2-Qualitative_Evaluation_PCA`](/4-qualitative_evaluation/4.2-Qualitative_Evaluation_PCA.ipynb) - Now that we've trained the fault detection classifier, we want to check its generability on the official test set. During training, we used the holdout method to select the test set as a subset of the data set, but remember: *this data set was the training set of the challenge to begin with*. Therefore, since its samples come from 4 different engines, but their identity is unknown, there is inevitably a redundancy between the selected test set and the train set. That said, the challenge website also offers a validation and a test set, unfortunately unlabelled to date. They contain $21.436$ samples each, for a total of $42.872$ samples, all recorded from 3 different engines than the 4 from which the train set was taken.
   - [`4.3-Qualitative_Evaluation_t-SNE`](/4-qualitative_evaluation/4.3-Qualitative_Evaluation_t-SNE.ipynb) - t-SNE is way better than PCA in preserving local structures.
However, *this test was aborted due to its complexity*. On my i7-10750H it would require about 1h!
   - [`4.4-KNN_In_PCA_Domain`](/4-qualitative_evaluation/4.4-KNN_In_PCA_Domain.ipynb) - The KNN uses the Euclidean distance between the test points and the labelled train points to make a prediction. Therefore, in this analysis we're asking if the test samples predicted as faulty by the PyKAN are near to faulty train samples.
