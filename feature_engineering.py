import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PATH_DATA_COMMIT = './data/init'
PATH_DATA_TOPIC = './results/topic_extraction'
PATH_DATA_SA = './results/sentiment_analysis'
PATH_RESULTS = './results/feature_engineering'

def commits_feature_engineering():
    """
    Function to execute the feature engineering step on commits data. 
    After selecting the dataset and the columns, have been grouped by author the data applying the mean or the sum as aggregation function.
    Finally, is computed the correlation matrix and selected the less redundant features. 
    """
    commits = pd.read_csv(os.path.join(PATH_DATA_COMMIT, 'commits_df.csv'))
    users = commits[['author','ns','nm','nf','entropy','la','ld','lt','ndev','pd','npt','exp','rexp','sexp']].groupby('author').mean()
    users[['ns_sum','nm_sum','nf_sum','la_sum', 'ld_sum','lt_sum','fix_sum','exp_sum','bug_sum']] = commits[['author','ns','nm','nf','la','ld','lt','fix','exp','bug']].groupby('author').sum()
    users['num_proj'] = commits[['author','repo']].drop_duplicates().groupby('author')['repo'].nunique()
    users['num_commit'] = commits[['author','commit']].groupby('author').count()

    column_order = ['ns','ns_sum','nm','nm_sum','nf','nf_sum',
                    'entropy', 'la','la_sum', 'ld', 'ld_sum', 'lt','lt_sum',
                    'fix_sum','ndev', 'pd', 'npt','exp','exp_sum','rexp', 
                    'sexp','bug_sum', 'num_commit','num_proj']
    users = users[column_order]

    correlation_matrix = users.loc[:, ~users.columns.isin(['nf','nf_sum','lt_sum','nm_sum','ns_sum','ld_sum','la_sum','exp','entropy'])].corr()

    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig(os.path.join(PATH_RESULTS, 'corr_commit.pdf') ,bbox_inches = 'tight')
    #plt.show()

    users_data = users.loc[:, ~users.columns.isin(['nf','nf_sum','lt_sum','nm_sum','ns_sum','ld_sum','la_sum','exp','entropy'])]
    users_data.to_csv(os.path.join(PATH_RESULTS, 'user_features_commits.csv'))


def sentiment_feature_engineering():
    """
    Function to execute the feature engineering step on sentiment analysis results. 
    After selecting the dataset of the best configuration, have been grouped by author and analyzed several possible features.
    Finally, is computed the correlation matrix and selected the less redundant features. 
    """
    sa_data = pd.read_csv(os.path.join(PATH_DATA_SA, 'issues_preprocessed_with_sentiment_finetuned.csv')).dropna()
    sa_data['sentiment'] = sa_data['sentiment'].replace({'neutral': 0, 'negative': -1, 'positive': 1})
    sa_features = sa_data.groupby('user.login')['sentiment'].mean().reset_index()
    sa_features.rename(columns={'sentiment': 'avg_sentiment'}, inplace=True)
    df_counts = sa_data.groupby('user.login')['sentiment'].agg(
        positive_count=lambda x: (x == 1).sum(),
        total_count='count'
    ).reset_index()

    df_counts['positive_ratio'] = df_counts['positive_count'] / df_counts['total_count']
    sa_features['positive_ratio'] = df_counts['positive_ratio']
    sa_features['comment_count'] = df_counts['total_count']

    sa_data['comment_length'] = sa_data['comment_body'].apply(len)
    df_comment_length = sa_data.groupby('user.login')['comment_length'].mean().reset_index(name='avg_comment_length')
    sa_features['avg_comment_length'] = df_comment_length['avg_comment_length']

    score_df = sa_data.groupby('user.login')['sentiment'].sum().reset_index().sort_values(by='sentiment', ascending=False)
    score_df.rename(columns={'sentiment': 'sentiment_sum'}, inplace=True)
    sa_features['sentiment_sum'] = score_df['sentiment_sum']

    sa_features.set_index('user.login',inplace=True)
    sa_features.to_csv(os.path.join(PATH_RESULTS, 'user_features_sa.csv'))
    correlation_matrix = sa_features.corr()

    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig(os.path.join(PATH_RESULTS,'correlazione_sa.pdf'),bbox_inches = 'tight')
    #plt.show()


def topic_feature_engineering():
    """
    Function to execute the feature engineering step on topic extraction results. 
    After selecting the dataset of the best configuration, have been grouped by author and analyzed several possible features.
    Finally, is computed the correlation matrix and selected the less redundant features. 
    """
    users_topic = pd.read_csv(os.path.join(PATH_DATA_TOPIC, 'issue_df_preprocessed_with_topics.csv'))
    users_topic.drop('Unnamed: 0',axis=1,inplace=True)
    users_topic_features = pd.DataFrame()
    users_topic_features = users_topic.groupby(['issue_user','topic']).size().reset_index(name='frequenza').pivot_table(index='issue_user',columns='topic',values='frequenza',fill_value=0)
    users_topic_features['unique_topics'] = users_topic.groupby('issue_user')['topic'].nunique()
    users_topic_features['avg_comments_issue'] = users_topic.groupby('issue_user')['num_comments'].mean()
    users_topic_features['total_issues'] = users_topic.groupby('issue_user').size()
    for topic in range(-1,10):
        users_topic_features[f'rel_freq_{topic}'] = users_topic_features[topic] / users_topic_features['total_issues']
    correlation_matrix = users_topic_features.corr()

    plt.figure(figsize=(20, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.savefig(os.path.join(PATH_RESULTS,'correlazione_topic.pdf'),bbox_inches = 'tight')
    #plt.show()

    users_topic_features = users_topic_features.drop(['-1','0','1','2','3','4','5','6','7','8','9'], axis=1)
    users_topic_features.to_csv(os.path.join(PATH_RESULTS,'users_topic_features.csv'))


def all_features_feature_engineering():
    """
    Function to perform the feature engineering task on the aggregation of all the features.
    After merging the less redundant features from the previous functions, is applied a group by author and visualized the correlation between all features.
    """
    df_commit = pd.read_csv(os.path.join(PATH_RESULTS, 'user_features_commits.csv'))
    df_sa = pd.read_csv(os.path.join(PATH_RESULTS, 'user_features_sa.csv'))
    df_topic = pd.read_csv(os.path.join(PATH_RESULTS, 'users_topic_features.csv'))

    merged_df = pd.merge(df_commit, df_sa, left_on='author', right_on='user.login', how='outer')

    merged_df['author'] = merged_df['author'].fillna(merged_df['user.login'])
    merged_df.drop(columns=['user.login'], inplace=True)

    merged_df = pd.merge(merged_df, df_topic, left_on='author', right_on='issue_user', how='outer')
    merged_df['author'] = merged_df['author'].fillna(merged_df['issue_user'])
    merged_df.drop(columns=['issue_user'], inplace=True)

    df_all = merged_df.fillna(0).set_index('author')
    correlation_matrix = df_all.corr()

    plt.figure(figsize=(15, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1,annot_kws={"fontsize":32})
    plt.savefig(os.path.join(PATH_RESULTS, 'correlazione_all.pdf'),bbox_inches = 'tight')
    #plt.show()
    
    df_all.to_csv(os.path.join(PATH_RESULTS, 'all_features.csv'))


def feature_engineering_step():
    """
    Function to execute all the procedures of the feature engineering pipeline step.
    """
    print('COMMIT')
    commits_feature_engineering()
    print('TOPIC')
    topic_feature_engineering()
    print('SENTIMENT')
    sentiment_feature_engineering()
    print('ALL')
    all_features_feature_engineering()

