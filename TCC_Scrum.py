#%% Instalando os pacotes necessários

!pip install pandas
!pip install numpy
!pip install statsmodels
!pip install matplotlib
!pip install seaborn
!pip install pingouin
!pip install statstests
!pip install scipy

#%% Importando os pacotes

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pingouin as pg
from statstests.process import stepwise

#%% Importando o banco de dados

respostas = pd.read_excel("Dados_MBA.xlsx")

# Variáveis métricas
respostas[['Scrum', 'Kanban']].describe()

# Variáveis categóricas
respostas[['Papel']].value_counts()
respostas[['Metodologia']].value_counts()

# Limpeza dos dados
respostas_scrum = respostas.drop(columns=['Kanban', 'Kanban_Medidor_Velocidade', 'Kanban_Troca_Prioridade'])
dados = respostas_scrum.drop(respostas_scrum[respostas_scrum.Metodologia != 'Scrum'].index)

dados[['Qualidade']].describe()

#%% Regressão Linear Simples (OLS)

# Análise gráfica da relação 1

sns.regplot(data=dados, x='Scrum_Objetivo_Definido', y='Feature_Release', ci=False, line_kws={'color':'red', 'lw':1})
plt.xlabel('Frequencia em que o objetivo da Sprint é definido (1 a 5)', fontsize=10)
plt.ylabel('Frequencia de Release (1 a 5)', fontsize=10)
plt.show()

# Análise gráfica da relação 2

sns.regplot(data=dados, x='Scrum_Troca_Prioridade', y='Feature_Release', ci=False, line_kws={'color':'red', 'lw':1})
plt.xlabel('Frequencia em que há alteração no escopo da Sprint atual (1 a 5)', fontsize=10)
plt.ylabel('Frequencia de Release (1 a 5)', fontsize=10)
plt.show()

# Análise gráfica da relação 3

sns.regplot(data=dados, x='Scrum_Medidor_Velocidade', y='Feature_Release', ci=False, line_kws={'color':'red', 'lw':1})
plt.xlabel('Frequencia em que o time olha para o Burndown (1 a 5)', fontsize=10)
plt.ylabel('Frequencia de Release (1 a 5)', fontsize=10)
plt.show()

# Análise gráfica da relação 4

sns.regplot(data=dados, x='Scrum_Plano_Retro', y='Feature_Release', ci=False, line_kws={'color':'red', 'lw':1})
plt.xlabel('Frequencia em que o time prioriza plano de ações definidos em Retro (1 a 5)', fontsize=10)
plt.ylabel('Frequencia de Release (1 a 5)', fontsize=10)
plt.show()

# Análise do coeficiente de correlação de Pearson da relação 1

pg.rcorr(respostas[['Feature_Release', 'Scrum_Objetivo_Definido']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

# Análise do coeficiente de correlação de Pearson da relação 2

pg.rcorr(respostas[['Feature_Release', 'Scrum_Troca_Prioridade']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

# Análise do coeficiente de correlação de Pearson da relação 3

pg.rcorr(respostas[['Feature_Release', 'Scrum_Medidor_Velocidade']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

# Análise do coeficiente de correlação de Pearson da relação 4

pg.rcorr(respostas[['Feature_Release', 'Scrum_Plano_Retro']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Regressão Linear Múltipla

# Estimação do modelo
feature_demo = sm.OLS.from_formula(
    formula = 'Feature_Demo ~ Scrum_Objetivo_Definido + Scrum_Objetivo_Completo + Scrum_Troca_Prioridade + Scrum_Medidor_Velocidade + Scrum_Estoria_Completa + Scrum_Plano_Retro',
    data=dados).fit()
feature_release = sm.OLS.from_formula(
    formula = 'Feature_Release ~ Scrum_Objetivo_Definido + Scrum_Objetivo_Completo + Scrum_Troca_Prioridade + Scrum_Medidor_Velocidade + Scrum_Estoria_Completa + Scrum_Plano_Retro',
    data=dados).fit()
team_performance = sm.OLS.from_formula(
    formula = 'Performance ~ Scrum_Objetivo_Definido + Scrum_Objetivo_Completo + Scrum_Troca_Prioridade + Scrum_Medidor_Velocidade + Scrum_Estoria_Completa + Scrum_Plano_Retro',
    data=dados).fit()
qualidade_scrum = sm.OLS.from_formula(
    formula = 'Qualidade ~ Scrum_Objetivo_Definido + Scrum_Objetivo_Completo + Scrum_Troca_Prioridade + Scrum_Medidor_Velocidade + Scrum_Estoria_Completa + Scrum_Plano_Retro',
    data=dados).fit()

# Obtenção dos outputs
feature_demo.summary()
feature_release.summary()
team_performance.summary()
qualidade_scrum.summary()

#%% Aplicando o stepwise ao modelo

# Procedimento stepwise para a remoção de variáveis não significativas
demo = stepwise(feature_demo, pvalue_limit=0.1)
release = stepwise(feature_release, pvalue_limit=0.1)
performance = stepwise(team_performance, pvalue_limit=0.1)
qualidade = stepwise(qualidade_scrum, pvalue_limit=0.1)

demo.summary()
release.summary()
performance.summary()
qualidade.summary()

#%% Fim!