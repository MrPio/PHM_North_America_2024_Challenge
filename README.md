# PHM North America 2024 Challenge
Addressing [PHM North America 2024 Challenge](https://data.phmsociety.org/phm2024-conference-data-challenge/) with [Kolmogorov-Arnold Networks](https://github.com/KindXiaoming/pykan).


## 📙 Slideshows

<img height="150rem" align="right" src="https://github.com/user-attachments/assets/9d513014-9f84-4387-b68e-7c7941bad89d"/>

### 1️⃣ First review [`.PPTX`](https://github.com/user-attachments/files/19043133/Gruppo.A1.prima.revisione.pptx) [`.PDF`](https://github.com/user-attachments/files/19043132/Gruppo.A1.prima.revisione.pdf)
### 2️⃣ Second review [`.PPTX`](https://github.com/user-attachments/files/19074287/Gruppo.A1.seconda.revisione.pptx) [`.PDF`](https://github.com/user-attachments/files/19074286/Gruppo.A1.seconda.revisione.pdf)
### 3️⃣ Third review [`.PPTX`](https://github.com/user-attachments/files/19250163/Gruppo.A1.terza.revisione.pptx) [`.PDF`](https://github.com/user-attachments/files/19250162/Gruppo.A1.terza.revisione.pdf)

## 📙 How this repository is structured

> [!IMPORTANT]  
> Before running the notebooks, make sure to extract the [`\dataset\0-original\dataset.zip`](/dataset/0-original/dataset.zip) archive. Then, you can begin running the notebooks in sequence, starting from [1.1](1-data_preprocessing/1.1-Exploratory_Data_Analysis.ipynb).

<p align="center">
    <img width="65%" src="img/strategy_map.png"/>
</p>

## 1️⃣ EDA:

#### PCA
<p align="center">
  <img width="600rem" src="https://github.com/user-attachments/assets/f187c558-0d2a-459e-93b0-590d87b243e8"/>
</p>

## 2️⃣ Probabilistic Regression:

#### KANs are interpretable
<p align="center">
  <img align="left" width="42%" src="https://github.com/user-attachments/assets/ba3aad37-ce1f-483b-b8e4-b8a209c415ca"/>
  <img width="42%" src="https://github.com/user-attachments/assets/16cfdd27-ec95-48e7-bed3-8b778fe10fc0"/>
</p>

#### Torque Target Probabilistic Regression with MLP, non-interpretable. GaussianNLL= -5
<p align="center">
  <img width="550rem" src="https://github.com/user-attachments/assets/c83f614f-3677-4739-b490-df943f152c03"/>
</p>

#### Torque Target Probabilistic Regression with [PyKAN](https://github.com/KindXiaoming/pykan), interpretable. GaussianNLL= -3
<p align="center">
  <img width="450rem" src="https://github.com/user-attachments/assets/47781fa4-12c6-4f9f-8a18-1e90eb24365f"/>
</p>

## 3️⃣ Fault Detection:

#### Binary classification with BCEWithLogitsLoss(pos_weight=2) = 0.18
<p align="center">
  <img align="left" width="42%" src="https://github.com/user-attachments/assets/0b95a142-294a-4777-b67b-ded899e49ee8"/>
  <img width="42%" src="https://github.com/user-attachments/assets/8f038a37-2efe-4d33-9cf3-e63c70209329"/>
</p>
