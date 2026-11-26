# Credit Card Fraud Detection

Projeto de análise de dados e Machine Learning desenvolvido para identificar transações financeiras fraudulentas por meio de algoritmos de classificação, com foco na avaliação de modelos em cenários de classes altamente desbalanceadas.

A proposta consiste em comparar diferentes algoritmos, avaliar sua capacidade de identificar fraudes e utilizar técnicas de interpretabilidade para compreender os fatores que influenciam as previsões.

## Objetivos

- Desenvolver modelos de classificação para detecção de fraudes financeiras.
- Comparar diferentes algoritmos de Machine Learning.
- Priorizar a identificação de transações fraudulentas por meio da métrica Recall.
- Avaliar o impacto do desbalanceamento das classes no desempenho dos modelos.
- Aplicar SHAP para interpretar as previsões do modelo selecionado.

## Tecnologias utilizadas

- Python
- Pandas e NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Jupyter Notebook

## 1. Problema de negócio

A detecção de fraudes financeiras é um problema de classificação binária no qual o objetivo é identificar transações fraudulentas (classe 1) e legítimas (classe 0).

O principal desafio do conjunto de dados é o forte desbalanceamento entre as classes.

No conjunto de teste, foram identificados:

| Classe | Quantidade | Representatividade |
|---|---:|---:|
| Transações legítimas (0) | 56.864 | 99,83% |
| Transações fraudulentas (1) | 98 | 0,17% |
| Total | 56.962 | 100% |

Esse desbalanceamento exige uma abordagem de avaliação diferente da utilizada em problemas tradicionais de classificação.

A acurácia, isoladamente, não representa adequadamente a capacidade de identificação de fraudes. Um modelo que classificasse todas as transações como legítimas poderia alcançar aproximadamente 99,83% de acurácia, sem identificar nenhuma fraude.

Por esse motivo, o Recall da classe positiva foi adotado como principal métrica de avaliação, acompanhado da Precisão e do F1-score.

## 2. Preparação dos dados

Inicialmente, foram separadas as variáveis explicativas (X) e a variável-alvo (y).

- **X:** atributos utilizados para identificar padrões nas transações.
- **y:** variável `Class`, responsável por indicar se uma transação é legítima ou fraudulenta.

A coluna `Class` foi removida das variáveis explicativas para evitar que a resposta fosse utilizada como entrada dos modelos.

A divisão dos dados foi realizada utilizando `train_test_split`, com os seguintes parâmetros:

- 80% para treinamento.
- 20% para teste.
- `random_state=42` para reprodutibilidade.
- `stratify=y` para preservar a proporção entre as classes.

### Tratamento do desbalanceamento

Foram utilizadas estratégias específicas de ponderação das classes:

- Regressão Logística: `class_weight='balanced'`.
- Random Forest: `class_weight='balanced'`.
- XGBoost: `scale_pos_weight`, calculado a partir da proporção entre as classes no conjunto de treinamento.

Essas estratégias permitem atribuir maior importância aos exemplos da classe minoritária durante o treinamento.

### Padronização

Na Regressão Logística, foi utilizado `StandardScaler` dentro de um Pipeline, garantindo que a padronização fosse ajustada exclusivamente com os dados de treinamento.

## 3. Modelagem

Foram desenvolvidos e comparados três algoritmos de classificação.

### 3.1 Regressão Logística

Modelo utilizado como abordagem inicial de classificação binária.

Configurações:
- `class_weight='balanced'`
- `max_iter=1000`
- `random_state=42`
- Padronização com `StandardScaler`.

### 3.2 Random Forest

Algoritmo baseado em um conjunto de árvores de decisão, utilizado para identificar relações e padrões não lineares nos dados.

Configurações:
- `n_estimators=50`
- `max_depth=8`
- `min_samples_split=5`
- `min_samples_leaf=2`
- `class_weight='balanced'`
- `random_state=42`

### 3.3 XGBoost

Algoritmo baseado em Gradient Boosting, utilizado para comparar o desempenho de uma abordagem de aprendizado por conjuntos de árvores.

Configurações:
- `n_estimators=200`
- `max_depth=5`
- `learning_rate=0.05`
- `scale_pos_weight` calculado a partir dos dados de treinamento.
- `eval_metric='logloss'`
- `random_state=42`

## 4. Avaliação e comparação dos modelos

Os modelos foram avaliados utilizando Precision, Recall e F1-score, considerando principalmente a classe 1 (fraudes).

### Resultados obtidos

| Modelo | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Regressão Logística | 0.06 | **0.92** | 0.11 |
| Random Forest | **0.64** | 0.86 | **0.73** |
| XGBoost | 0.52 | 0.86 | 0.65 |

### Análise dos resultados

**Regressão Logística**

Apresentou o maior Recall, de 0.92, demonstrando capacidade de identificar aproximadamente 92% das fraudes presentes no conjunto de teste.

Entretanto, sua Precisão de 0.06 indica uma quantidade elevada de falsos positivos, classificando muitas transações legítimas como suspeitas.

**Random Forest**

Apresentou o melhor equilíbrio entre as métricas, alcançando:

- Precision: 0.64
- Recall: 0.86
- F1-score: 0.73

O modelo conseguiu identificar aproximadamente 86% das fraudes, mantendo uma precisão significativamente superior à dos demais algoritmos.

**XGBoost**

Apresentou Recall de 0.86, igual ao Random Forest, porém com Precisão de 0.52 e F1-score de 0.65.

### Modelo selecionado

O Random Forest foi considerado o modelo com melhor equilíbrio entre identificação de fraudes e controle de falsos positivos entre os três algoritmos avaliados.

Embora a Regressão Logística tenha apresentado maior Recall, sua baixa Precisão representa uma limitação operacional importante.

## 5. Limiar de decisão (Threshold)

Para a classificação final, foi utilizado o limiar padrão de decisão de 0.5, disponibilizado pelo método `predict()` dos classificadores.

Nesse cenário, probabilidades iguais ou superiores a 0.5 são classificadas como fraudes, enquanto probabilidades inferiores são classificadas como transações legítimas.

Não foi realizado um processo adicional de otimização do threshold nesta versão do projeto.

O ajuste desse limiar representa uma possibilidade de evolução futura, permitindo investigar diferentes relações entre Recall, Precisão e falsos positivos.

## 6. Interpretabilidade com SHAP

Para compreender o comportamento do Random Forest, foi utilizada a biblioteca SHAP (SHapley Additive exPlanations).

A análise foi realizada utilizando uma amostra de 2.000 registros do conjunto de teste, considerando as contribuições das variáveis para a previsão da classe positiva (fraudes).

Foram utilizados dois tipos de visualização.

### 6.1 Feature Importance

O gráfico de importância média absoluta dos valores SHAP identificou as seguintes variáveis como as mais relevantes:

1. V14
2. V12
3. V4
4. V10
5. V11
6. V3
7. V17

As variáveis V14, V12 e V4 apresentaram as maiores contribuições médias absolutas para as previsões do modelo.

### 6.2 Summary Plot (Beeswarm)

O gráfico Beeswarm permitiu analisar a direção das contribuições das variáveis, relacionando seus valores com o impacto nas previsões.

Entre os principais padrões observados:

- **V14:** valores baixos tendem a contribuir positivamente para a previsão de fraude.
- **V12:** valores baixos também apresentam tendência de aumentar a previsão de fraude.
- **V4:** valores altos tendem a contribuir positivamente para a previsão de fraude.
- **V10:** valores altos tendem a contribuir negativamente para a previsão de fraude.
- **V11:** valores altos tendem a contribuir positivamente para a previsão de fraude.

A análise demonstra que o modelo concentra boa parte de sua capacidade preditiva em determinadas variáveis, permitindo compreender melhor os padrões utilizados durante a classificação.

É importante destacar que os valores SHAP representam a contribuição das variáveis para as previsões do modelo, não estabelecendo relações de causalidade.

## 7. Diferenciais da implementação

Nesta implementação, foram exploradas diferentes abordagens para lidar com o desbalanceamento e melhorar a análise dos resultados:

- Comparação entre três algoritmos de classificação.
- Aplicação de estratégias de ponderação das classes.
- Utilização de Pipeline para padronização na Regressão Logística.
- Comparação conjunta de Precision, Recall e F1-score.
- Análise de probabilidades de classificação.
- Utilização de SHAP para interpretação das previsões.

Essas decisões permitiram avaliar não apenas a capacidade de identificação de fraudes, mas também o equilíbrio entre detecção e falsos positivos.

## 8. Conclusão

O projeto demonstrou a importância de selecionar métricas adequadas para problemas de classificação com forte desbalanceamento de classes.

A comparação evidenciou que um modelo com maior Recall não necessariamente apresenta o melhor desempenho operacional, sendo necessário considerar também a Precisão e o F1-score.

Entre os algoritmos avaliados, o Random Forest apresentou o melhor equilíbrio, com Recall de 0.86 e F1-score de 0.73.

A utilização do SHAP complementou a avaliação ao permitir identificar as variáveis que mais influenciaram as previsões.

Como possibilidades de evolução, destacam-se:

- Otimização de hiperparâmetros.
- Ajuste do threshold de classificação.
- Avaliação de técnicas adicionais para tratamento do desbalanceamento.
- Comparação de diferentes estratégias de validação.
- Análise mais aprofundada dos falsos positivos e falsos negativos.

O trabalho reforça a importância de combinar modelagem, avaliação estatística e interpretabilidade na construção de soluções de Machine Learning aplicadas a problemas reais.

## Estrutura do repositório

```text
credit-card-fraud-detection/
│
├── notebooks/
│   └── fraud_detection_analysis.ipynb
│
├── images/
│   ├── shap_feature_importance.png
│   └── shap_summary_plot.png
│
├── README.md
└── requirements.txt
```

## Como executar

Clone o repositório:

```bash
git clone https://github.com/rodrigoambar/credit-card-fraud-detection.git
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Abra o notebook em um ambiente Jupyter ou google colab e disponibilize o conjunto de dados no caminho esperado pelo notebook.

---

**Projeto desenvolvido como parte de um desafio prático de Python, Análise de Dados e Machine Learning.**
