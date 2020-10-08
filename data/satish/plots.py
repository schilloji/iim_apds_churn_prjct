def outlier_plot(df,low_cut,high_cut):
    #Box plot to under stand the ouliers
    
    numeric_df=df.select_dtypes(exclude='object')
    numeric_df=numeric_df[numeric_df.columns.drop(list(['churn','Customer_ID']))]
    numeric_col=numeric_df.columns[:10]
    #sns.distplot(df[(df[column]>df[column].quantile(low_cut))&
    #          (df[column]<df[column].quantile(high_cut))][column].dropna(),ax=axes[1])
    
    fig, axes = plt.subplots(round(len(numeric_col) / 3), 2, figsize=(12, 9))
    
    for i, ax in enumerate(fig.axes):
        if i < len(numeric_col):
            #ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            sns.boxplot(numeric_df[(numeric_df[numeric_col[i]]>numeric_df[numeric_col[i]].quantile(low_cut))&
              (numeric_df[numeric_col[i]]<numeric_df[numeric_col[i]].quantile(high_cut))][numeric_col[i]].dropna(),ax=ax)

            #sns.countplot(x=telo_df_cat.columns[i],hue='churn', alpha=0.7, data=telo_df_cat, ax=ax)
        #fig.tight_layout()


outlier_plot(telo_df,0.05,0.95)



#Exploration of categorical varaible
categ_col = telo_df.select_dtypes(include = ['object'])
telo_df_cat=pd.concat([categ_col,telo_df['churn']],axis=1)
telo_df_cat.head()

fig, axes = plt.subplots(round(len(telo_df_cat.columns) / 3), 2, figsize=(12, 30))

for i, ax in enumerate(fig.axes):
    if i < len(telo_df_cat.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        sns.countplot(x=telo_df_cat.columns[i],hue='churn', alpha=0.7, data=telo_df_cat, ax=ax)

fig.tight_layout()
