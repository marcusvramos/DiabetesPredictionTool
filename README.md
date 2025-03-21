# Previsão de Diabetes com Streamlit e TensorFlow

Este projeto usa uma rede neural MLP (TensorFlow) para prever diabetes com base em dados do paciente, exibindo os resultados em uma interface Streamlit.

## Pré-requisitos

- Python 3.11 (recomendado)
- Ambiente virtual (opcional, mas sugerido)

## Passo a passo para rodar o projeto

 1. Criar um ambiente virtual
Crie um ambiente virtual para isolar as dependências:

```bash
python3.11 -m venv myenv-tf
```

### 2. Ativar o ambiente virtual

Ative o ambiente com:

Linux/Mac:

```bash
source myenv-tf/bin/activate
```

### 3. Instalar as dependências

Instale as dependencias:

```bash
pip install tensorflow streamlit joblib scikit-learn pandas numpy
```

### 4. Rodar a aplicação

Execute o comando abaixo para iniciar a interface Streamlit:

```bash
python -m streamlit run app.py
```