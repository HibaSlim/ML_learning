import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('Market_Basket_Optimisation.csv')
df.head()
df.info()
df['Product line'].value_counts()
df['Branch'].value_counts()
df['Payment'].value_counts()
transactions = df.groupby('Invoice ID')['Product line'].apply(list).tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
frequent_itemsets.sort_values(by='support', ascending=False)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.sort_values(by='lift', ascending=False)
rules.head()
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]



top_rules = rules.nlargest(10, 'lift')
fig = px.bar(top_rules, x=top_rules.index, y='lift',
             hover_data=['antecedents', 'consequents', 'confidence'],
             title="Top 10 Association Rules by Lift")
fig.show()