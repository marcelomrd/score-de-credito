import pandas as pd
import numpy as np
import re
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

from collections import Counter

sns.set(style='white', context='notebook', palette='deep')
pd.options.display.max_columns = None


treino = pd.read_csv("treino.csv")
teste = pd.read_csv("teste.csv")

treino.isnull().sum()
teste.isnull().sum()

ax = sns.countplot(x = treino.inadimplente ,palette="Set3")
sns.set(font_scale=1.5)
ax.set_ylim(top = len(treino))
ax.set_xlabel('Inadimplência')
ax.set_ylabel('Frequência')
fig = plt.gcf()
fig.set_size_inches(10,5)

plt.show()

count_no_default = len(treino[treino['inadimplente']==0])
count_default = len(treino[treino['inadimplente']==1])
pct_of_no_default = count_no_default/(count_no_default+count_default)
print("Porcentagem de adimplentes", pct_of_no_default*100,"%")
pct_of_default = count_default/(count_no_default+count_default)
print("Porcentagem de inadimplentes", pct_of_default*100,"%")

def detect_outliers(df,n,features):
    outlier_indices = []
    
    for col in features:

        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col],75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers

Outliers_to_drop = detect_outliers(treino,2,['util_linhas_inseguras',
                                            'idade',
                                            'vezes_passou_de_30_59_dias',
                                            'razao_debito',
                                            'salario_mensal',
                                            'numero_linhas_crdto_aberto',
                                            'numero_vezes_passou_90_dias',
                                            'numero_emprestimos_imobiliarios',
                                            'numero_de_vezes_que_passou_60_89_dias',
                                            'numero_de_dependentes'])

treino.loc[Outliers_to_drop]
treino = treino.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

tam_treino = len(treino)
dataset =  pd.concat(objs=[treino, teste], axis=0).reset_index(drop=True)

dataset = dataset.rename(columns={'util_linhas_inseguras':'inseguras',
                                'vezes_passou_de_30_59_dias':'atraso_30_59_dias',
                                'razao_debito':'r_debito',
                                'salario_mensal':'salario',
                                'numero_linhas_crdto_aberto':'linhas_crdto',
                                'numero_vezes_passou_90_dias':'atraso_90_dias',
                                'numero_emprestimos_imobiliarios':'emp_imob',
                                'numero_de_vezes_que_passou_60_89_dias':'atraso_60_89_dias',
                                'numero_de_dependentes':'dependentes',})

treino = treino.rename(columns={'util_linhas_inseguras':'inseguras',
                                'vezes_passou_de_30_59_dias':'atraso_30_59_dias',
                                'razao_debito':'r_debito',
                                'salario_mensal':'salario',
                                'numero_linhas_crdto_aberto':'linhas_crdto',
                                'numero_vezes_passou_90_dias':'atraso_90_dias',
                                'numero_emprestimos_imobiliarios':'emp_imob',
                                'numero_de_vezes_que_passou_60_89_dias':'atraso_60_89_dias',
                                'numero_de_dependentes':'dependentes',})

teste = teste.rename(columns={'util_linhas_inseguras':'inseguras',
                                'vezes_passou_de_30_59_dias':'atraso_30_59_dias',
                                'razao_debito':'r_debito',
                                'salario_mensal':'salario',
                                'numero_linhas_crdto_aberto':'linhas_crdto',
                                'numero_vezes_passou_90_dias':'atraso_90_dias',
                                'numero_emprestimos_imobiliarios':'emp_imob',
                                'numero_de_vezes_que_passou_60_89_dias':'atraso_60_89_dias',
                                'numero_de_dependentes':'dependentes',})

g = sns.heatmap(treino.corr(),annot=False, fmt = ".1f")

dataset.inseguras.describe()
dataset.inseguras = pd.qcut(dataset.inseguras.values, 5).codes

def bargraph(xaxis):
    
    g  = sns.factorplot(x=xaxis,y="inadimplente",data=dataset,kind="bar", size = 6 , 
    palette = "muted")
    g.despine(left=True)
    g = g.set_ylabels("Probabilidade de inadimplência")
    return g

g = bargraph("inseguras")

g = sns.FacetGrid(dataset, col='inadimplente')
g = g.map(sns.distplot, "idade")

dataset.idade = pd.qcut(dataset.idade.values, 5).codes

g = bargraph("idade")

g = bargraph("atraso_30_59_dias")

for i in range(len(dataset)):
    if dataset.atraso_30_59_dias[i] >= 6:
        dataset.atraso_30_59_dias[i] = 6

g = bargraph("atraso_30_59_dias")

g = sns.FacetGrid(dataset, col='inadimplente')
g = g.map(sns.distplot, "r_debito")

dataset.r_debito = pd.qcut(dataset.r_debito.values, 5).codes

g = bargraph("r_debito")

dataset.salario.isnull().sum()

g = sns.heatmap(dataset[["salario","inseguras","idade","r_debito","linhas_crdto"]].corr(),cmap="BrBG",annot=True)

g = sns.heatmap(dataset[["salario","emp_imob","dependentes"]].corr(),cmap="BrBG",annot=True)

g = sns.heatmap(dataset[["salario","atraso_30_59_dias","atraso_60_89_dias","atraso_90_dias"]].corr(),cmap="BrBG",annot=True)

dataset.salario.median()

dataset.salario = dataset.salario.fillna(dataset.salario.median())

dataset.salario = pd.qcut(dataset.salario.values, 5).codes

g = bargraph("salario")

dataset.linhas_crdto.describe()

dataset.linhas_crdto = pd.qcut(dataset.linhas_crdto.values, 5).codes

g = bargraph("linhas_crdto")

dataset.atraso_90_dias.describe()

g = bargraph("atraso_90_dias")

for i in range(len(dataset)):
    if dataset.atraso_90_dias[i] >= 5:
        dataset.atraso_90_dias[i] = 5

g = bargraph("atraso_90_dias")

dataset.emp_imob.describe()

g = bargraph("emp_imob")

for i in range(len(dataset)):
    if dataset.emp_imob[i] >= 6:
        dataset.emp_imob[i] = 6

g = bargraph("emp_imob")

g = bargraph("atraso_60_89_dias")

for i in range(len(dataset)):
    if dataset.atraso_60_89_dias[i] >= 3:
        dataset.atraso_60_89_dias[i] = 3

g = bargraph("atraso_60_89_dias")

dataset.dependentes.isnull().sum()

dataset.dependentes = dataset.dependentes.fillna(dataset.dependentes.median())

g = bargraph("dependentes")

for i in range(len(dataset)):
    if dataset.dependentes[i] >= 4:
        dataset.dependentes[i] = 4

g = bargraph("dependentes")

dataset = pd.get_dummies(dataset, columns = ["inseguras"], prefix="inseguras")
dataset = pd.get_dummies(dataset, columns = ["idade"], prefix="idade")
dataset = pd.get_dummies(dataset, columns = ["atraso_30_59_dias"], prefix="atraso_30_59_dias")
dataset = pd.get_dummies(dataset, columns = ["r_debito"], prefix="r_debito")
dataset = pd.get_dummies(dataset, columns = ["salario"], prefix="salario")
dataset = pd.get_dummies(dataset, columns = ["linhas_crdto"], prefix="linhas_crdto")
dataset = pd.get_dummies(dataset, columns = ["atraso_90_dias"], prefix="atraso_90_dias")
dataset = pd.get_dummies(dataset, columns = ["emp_imob"], prefix="emp_imob")
dataset = pd.get_dummies(dataset, columns = ["atraso_60_89_dias"], prefix="atraso_60_89_dias")
dataset = pd.get_dummies(dataset, columns = ["dependentes"], prefix="dependentes")

treino = dataset[:tam_treino]
teste = dataset[tam_treino:]
teste.drop(labels=["inadimplente"],axis = 1,inplace=True)

treino["inadimplente"] = treino["inadimplente"].astype(int)

Y_train = treino["inadimplente"]

X_train = treino.drop(labels = ["inadimplente"],axis = 1)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X_train, Y_train)

features = pd.DataFrame()
features['Variável'] = X_train.columns
features['Importância'] = clf.feature_importances_
features.sort_values(by=['Importância'], ascending=True, inplace=True)
features.set_index('Variável', inplace=True)

features.plot(kind='barh', figsize=(20, 20));

parameters = {'n_estimators': 1000, 'random_state' : 20}
    
model = RandomForestClassifier(**parameters)
model.fit(X_train, Y_train)

DefaultProba = model.predict_proba(teste)
DefaultProba = DefaultProba[:,1]

saving_data = pd.read_csv("teste.csv")
saving_data["Prob"] = DefaultProba
saving_data["inadimplente"] = 0

threshold = 0.1
pct = 20

while (pct > 7):
    threshold = threshold + 0.01
    saving_data["inadimplente"] = 0
    saving_data["inadimplente"][saving_data["Prob"]>threshold] = 1
    pct = saving_data["inadimplente"].sum()/len(saving_data)*100
threshold,pct

del saving_data['Prob']
saving_data.to_csv("teste.csv")