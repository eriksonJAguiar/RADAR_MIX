# RADAR-MIX: How to Uncover Adversarial Attacks in Medical Image Analysis through Explainability

This repository comprehends the source code developed in the **RADAR-MIX** proposal.

**Authors:** Erikson J. de Aguiar, Caetano Traina Junior, and Agma J. M. Traina

Contents: [[Paper]](https://ieeexplore.ieee.org/abstract/document/10600758), [[Dataset]](https://challenge.isic-archive.com/data/#2018), [[Quickstart]](#Quickstart), [[Bibtex]](#Bibtex)

**Conference: IEEE 37th Symposium on Computer-Based Medical Systems (CBMS)**

**Summary of proposal:** Medical image analysis is an important asset in the clinical process, providing resources to assist physicians in detecting diseases and making accurate diagnoses. Deep Learning (DL) models have been widely applied in these tasks, improving the ability to recognize patterns, including accurate and fast diagnosis. However, DL can present issues related to security violations that reduce the system’s confidence. Uncovering these attacks before they happen and visualizing their behavior is challenging. Current solutions are limited to binary analysis of the problem, only classifying the sample into attacked or not attacked. In this paper, we propose the RADAR-MIX framework for uncovering adversarial attacks using quantitative metrics and analysis of the attack’s behavior based on visual analysis. The RADAR-MIX provides a framework to assist practitioners in checking the possibility of adversarial examples in medical applications. Our experimental evaluation shows that the Deep-Fool and Carlini & Wagner (CW) attacks significantly evade the ResNet50V2 with a slight noise level of 0.001. Furthermore, our results revealed that the gradient-based methods, such as Gradient-weighted Class Activation Mapping (Grad-CAM) and SHapley Additive exPlanations (SHAP), achieved high attack detection effectiveness. While Local Interpretable Model-agnostic Explanations (LIME) presents low consistency, implying the most ability to uncover robust attacks supported by visual analysis.

## Quickstart

To quickly get started with the **RADAR-MIX** repository, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/RADAR_MIX.git
    cd RADAR_MIX
    ```

2. Create and activate a Python environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

For more detailed instructions, refer to the [Setup](#Setup) section.

## Setup

### Run clean model

```python
python train_clean_model.py --dataset dataset_path --model_name resnet50 --epochs 10 --dataset_name my_dataset_name
```

### Run experiments with attacks

```python
python run_experiments_attacks_xai.py -dm dataset_name -d dataset_path -dv dataset_csv -wp weights_path
```


## Bibtex

If you use this repository, please cite the following paper:

```
@inproceedings{Aguiar2024,
  title = {RADAR-MIX: How to Uncover Adversarial Attacks in Medical Image Analysis through Explainability},
  url = {http://dx.doi.org/10.1109/CBMS61543.2024.00078},
  DOI = {10.1109/cbms61543.2024.00078},
  booktitle = {2024 IEEE 37th International Symposium on Computer-Based Medical Systems (CBMS)},
  publisher = {IEEE},
  author = {De Aguiar,  Erikson J. and Traina,  Caetano and Traina,  Agma J. M.},
  year = {2024},
  month = jun,
  pages = {436–441}
}
```

## Contact

For more information, please email me at **erjulioaguiar@usp.br** or text me on [LinkedIn](https://www.linkedin.com/in/erjulioaguiar/)

