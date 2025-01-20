import os
import pandas as pd
from tqdm import tqdm
import ast
import json
import re
from utils import fetch_gh_data, REPOS_MAPPING, JIT_EXTENSIONS

PATH_JIT = './data/JiTReliability'
PATH_RAW = './data/raw'
PATH_INIT = './data/init'

LINKS_PATTERNS = {
    'ATTACHMENTS': re.compile(r'https://apache\.github\.io/[A-Za-z]+-jira-archive/attachments/[A-Z0-9-]+'),
    'JIRA_ISSUE_COMMENT': re.compile(r'https:\/\/issues\.apache\.org\/jira\/browse\/[A-Z]+-\d+\?focusedCommentId=\d+&page=com\.atlassian\.jira\.plugin\.system\.issuetabpanels:comment-tabpanel#comment-\d+\)\)'),
      'JIRA_ISSUE' : re.compile(r'https://issues\.apache\.org/jira/browse/[A-Z0-9-]+')
}

tqdm.pandas()


def commits_id_formatting(main_df, data, root):
    """
    Adds infos to retrieve commits details from the single ID of a '***_commitsID.csv' file.
    :param main_df: the comprehensive dataframe of commits for all the JiT Reliability projects.
    :param data: the path of the '***_commitsID.csv' file to extract.
    :param root: the root of the directory analysed.
    :return: main_df with the new columns 'transactionid', 'repo', 'call'.
    """
    df_commitsID = pd.read_csv(data, sep='\t')
    df_commitsID['transactionid'] = os.path.basename(os.path.dirname(data)) + "_" + \
                                    df_commitsID['transactionid'].astype(str)
    df_commitsID['repo'] = REPOS_MAPPING.get(os.path.basename(root))
    df_commitsID['call'] = 'https://api.github.com/repos/' + df_commitsID['repo'] + '/commits/' + df_commitsID['commit']
    main_df = pd.concat([main_df, df_commitsID])
    return main_df


def bug_dataset_formatting(main_df, data_bug, data_bug_eb):
    """
    Aggregates the JiT Reliability features from the commits of '***_input.csv' and '***_inputEB.csv' files.
    :param main_df: the comprehensive dataframe of commits for all the JiT Reliability projects.
    :param data: the path of the '***_input.csv' file to extract.
    :param data: the path of the '***_inputEB.csv' file to extract.
    :return: main_df with the commits features from both datasets.
    """
    df_bug = pd.read_csv(data_bug)
    df_bug['transactionid'] = os.path.basename(os.path.dirname(data_bug)) + "_" + df_bug['transactionid'].astype(str)
    df_bug.set_index('transactionid', inplace=True)

    df_bug_EB = pd.read_csv(data_bug_eb)
    df_bug_EB.rename(columns={"bug": "bug_EB"}, inplace=True)
    df_bug_EB['transactionid'] = os.path.basename(os.path.dirname(data_bug_eb)) + "_" + df_bug_EB['transactionid'].astype(str)
    df_bug_EB.set_index('transactionid', inplace=True)
    
    cols_to_use = df_bug_EB.columns.difference(df_bug.columns)
    df_bug = pd.merge(df_bug, df_bug_EB[cols_to_use], left_index=True, right_index=True, how='inner')

    main_df = pd.concat([main_df, df_bug])
    return main_df


def issues_retrieve_formatting(data, root):
    """
    Adds infos to retrieve issues details from the single ID of a '***_failures.csv' file.
    :param data: the path of the '***_commitsID.csv' file to extract.
    :param root: the root of the directory analysed.
    :return: temp_df with the failures details of a single project with the columns 'transactionid', 'repo', 'call'.
    """
    df = pd.read_csv(data, sep='\t')
    temp_df = df[['issue.names']].copy()
    temp_df['issue.names'] = temp_df['issue.names'].astype(str)
    temp_df['repo'] = REPOS_MAPPING.get(os.path.basename(root))
    temp_df['issue_call'] = 'https://api.github.com/repos/' + temp_df['repo'] + '/issues/' + temp_df['issue.names'].apply(
        lambda x: x.split('-')[1] if '-' in x else x)
    return temp_df


def jit_extraction():
    """
    Function for JiT Reliability preprocessing. This function extracts two datasets:
    - 'commits.csv': from the merge of the '***_commitsID.csv' files and the '***_input.csv' + '***_inputEB.csv' files, 
                    contains the JiT Reliability features about commits with the bug categorization;
    - 'issues.csv': from the '***_failueres.csv' files.
    """
    commits_df = pd.DataFrame()
    bugs_df = pd.DataFrame()
    issues_dfs = []

    for root, _, files in os.walk(PATH_JIT):
        selected_files = {}
        for file in files:
            if file.endswith('csv'):
                for ext in JIT_EXTENSIONS:
                    if file.endswith(ext):
                        selected_files[ext] = os.path.join(root, file)

                if len(selected_files) == len(JIT_EXTENSIONS):
                    commits_df = commits_id_formatting(commits_df, selected_files["commitsID.csv"],root)
                    bugs_df = bug_dataset_formatting(bugs_df, selected_files["input.csv"], selected_files["input_EB.csv"])
                    issues_dfs.append(issues_retrieve_formatting(selected_files["failures.csv"], root))
                    break

    # 'commits_raw.csv' definition
    commits_bug_df = pd.merge(commits_df, bugs_df, on='transactionid', how='inner')
    commits_bug_df.drop_duplicates(inplace=True)
    commits_bug_df.drop('Unnamed: 0', axis=1,inplace=True)
    print('SHAPE DF COMMIT', str(commits_bug_df.shape))
    commits_bug_df.to_csv(os.path.join(PATH_RAW, 'commits_raw.csv'))
    
    # 'issues_raw.csv' definition
    issues_df = pd.concat(issues_dfs, ignore_index=True)
    print('SHAPE ISSUES', str(issues_df.shape))
    issues_df.to_csv(os.path.join(PATH_RAW, 'issues_raw.csv'))


def retrieve_commits_info():
    """
    Function to retrieve commits events details from the GitHub REST API. 
    The resultant 'commits_df.csv' is the updated version of 'commits_raw.csv' with the details retrieved with the column 'call'.
    """
    commits_df = pd.DataFrame()
    commits_raw = pd.read_csv(os.path.join(PATH_RAW, 'commits_raw.csv'))
    commits_df = commits_raw.copy()

    print(f'Fetching {commits_df.shape[0]} Commits ...')
    commits_df['details'] = commits_df['call'].progress_apply(fetch_gh_data)
    commits_df = commits_df[commits_df['details'] != 'NOT FOUND']
    commits_df = commits_df.loc[:, ~commits_df.columns.str.contains('^Unnamed')]

    commits_df.to_csv(os.path.join(PATH_INIT, 'commits_df.csv'), index=False)


def author_extraction(commits_df):
    """
    Function to extract from the "commits_df.csv"'s field 'details' the author login. 
    A commit could have two possibile users, one that is the 'author' and another one that is the 'committer', 
        for this analysis the author is considerent enough.
    :param commits_df: the commits' dataframe to enhance.
    :return commits_df: the commits' dataframe with the author login column.
    """    
    authors = []
    for details in tqdm(commits_df['details']):
        try:
            authors.append(pd.json_normalize(ast.literal_eval(details))['author.login'])
        except Exception:
            authors.append('N')
            pass
    
    logins = []
    for author in authors:
        logins.append(author[0])

    commits_df['author'] = pd.Series(logins)
    commits_df = commits_df[commits_df['author']!='Not Found']
    return commits_df 


def issues_extraction():
    """
    Function to retrieve, given a GitHub REST API call, the issues informations. 
        In particular, are extracted also the details about the comments associated to the issue.
        The result is 'issues_raw.csv', a dataframe with the json form result of each issue and their associated comments. 
    """
    issues_df = pd.read_csv(os.path.join(PATH_RAW, 'issues_raw.csv'))
    print(f'Fetching {issues_df.shape[0]} Issues ...')
    issues_df['issue'] = issues_df['issue_call'].apply(fetch_gh_data)
    issues_df.to_csv(os.path.join(PATH_INIT, 'issues_df.csv'))

    issues_df['num_comments'] = issues_df['issue'].apply(lambda x: x.get('comments'))
    issues_df['issue_title'] = issues_df['issue'].apply(lambda x: x.get('title'))
    issues_df['issue_body'] = issues_df['issue'].apply(lambda x: x.get('body'))
    issues_df['issue_user'] = issues_df['issue'].apply(lambda x: x.get('user').get('login'))
    issues_df['issue_labels'] = issues_df['issue'].apply(lambda x: x.get('label'))
    issues_df['issue_author_association'] = issues_df['issue'].apply(lambda x: x.get('author_association'))
    issues_df['issue_created_at'] = issues_df['issue'].apply(lambda x: x.get('created_at'))
    issues_df['issue_updated_at'] = issues_df['issue'].apply(lambda x: x.get('updated_at'))
    issues_df['issue_closed_at'] = issues_df['issue'].apply(lambda x: x.get('closed_at'))
    issues_df['is_pull_request'] = issues_df['issue'].apply(is_pull_request)

    print('Extracting comments ...')
    issues_df['comments_call'] = issues_df['issue'].apply(
        lambda x: x.get('comments_url') if x.get('comments', 0) > 0 else None)
    issues_df['comments'] = issues_df['comments_call'].apply(lambda x: fetch_gh_data(x) if x is not None else None)
    extracted_comments = issues_df['comments'].apply(
        lambda x: issues_comments_extraction(x) if x is not None else ([], [], [], []))
    extracted_comments_df = pd.DataFrame(extracted_comments.tolist(),
                                         columns=['comments_ids', 'comments_bodies', 'comments_users',
                                                  'comments_dates'])
    issues_df = issues_df.join(extracted_comments_df)
    print(issues_df.columns)
    issues_df.to_csv(os.path.join(PATH_RAW, 'issues_raw.csv'))


def is_pull_request(issue_str):
    """
    Function to evaluate if an issue is also a pull request or not. 
    This information gives more details on the usage of the issue event in the repository.
    :param issue_str: 'issue' column of the issues_df
    :return bool: True if the issue is a pull request, else False
    """
    try:
        issue_json = json.loads(json.dumps(issue_str))
        return 'pull_request' in issue_json
    except json.JSONDecodeError:
        return False


def issues_comments_extraction(comments):
    """
    Extract comments parameters from the results of a GitHub API in JSON format.
    :param comments:is the results of a api call for each comment in a dataframe
    :return: 4 lists, each of them with one feature of the comments.
            In particular, 
                - the comments_ids, 
                - the comments_bodies (the text), 
                - the 'comments_users' (who commented), and 
                - 'comments_dates' (when the comment has been uploaded)
    """
    comments_ids = []
    comments_bodies = []
    comments_users = []
    comments_dates = []
    for comment in comments:
        comments_ids.append(comment.get('id'))
        comments_bodies.append(comment.get('body'))
        comments_users.append(comment.get('user').get('login'))
        comments_dates.append(comment.get('updated_at'))
    return comments_ids, comments_bodies, comments_users, comments_dates


def issues_json_formatting():
    """
    Function to extract and process the raw issues columns in json format. 
    The result is a 'issues_df.csv' file with columns for each relevant attribute for the two events.
    """
    issues_df = pd.read_csv(os.path.join(PATH_RAW, 'issues_raw.csv'))
    dataframes = []

    def comments_json_unroll(comments):
        """
        Function to unroll comments features from the json resultant.
        :param comments: the json objects of the comments in a list.
        """
        if type(comments) is not float:
            comments = ast.literal_eval(comments)
            df = pd.json_normalize(comments)
            df = pd.DataFrame(df)
            dataframes.append(df)

    issues_df['comments'].apply(comments_json_unroll)
    comments_df = pd.concat(dataframes, ignore_index=True)
    
    
    columns_to_keep = ['issue_url','id','created_at','updated_at','author_association','body','user.login','reactions.total_count','reactions.+1', 'reactions.-1', 'reactions.laugh', 'reactions.hooray', 'reactions.confused', 'reactions.heart', 'reactions.rocket','reactions.eyes']
    comments_df = comments_df[columns_to_keep]

    issues_df = pd.merge(issues_df, comments_df, how='outer', left_on='issue_call', right_on='issue_url')
    issues_df.drop(columns=['issue_url','comments_ids','comments_bodies','comments_users','comments_dates','issue_labels'],inplace=True)
    issues_df.rename(columns={'updated_at_x':'issue_updated_at', 'updated_at_y': 'comment_updated_at','created_at': 'comment_created_at', 'body': 'comment_body'}, inplace=True)
    issues_df.drop(columns=['repo','issue_call','comments_call','issue','comments'],inplace=True)
    issues_df = issues_df.loc[:, ~issues_df.columns.str.contains('^Unnamed')]
    issues_df.to_csv(os.path.join(PATH_INIT, 'issues_df.csv'))


def split_strings(strings, separator):
    """
    Utils function to split the string using a given separator.
    :param strings: text to process.
    :param separator: str for the separator.
    :return an array with the splitted sentence for each element.
    """
    return [s.split(separator)[0] for s in strings], [s.split(separator)[1] if len(s.split(separator)) > 1 else None for s in strings]


def extract_pattern(text):
    """
    Utils function to extract the login string from another string.
    :param text: the 'asfimport' sentence.
    :return x: the author string, if in the input sentence
    """
    pattern_login = "\(\@[A-Za-z0-9]*\)"
    if type(text) is str:
        x = re.findall(pattern_login, text)
        if x :
            x = x[0]
            x = x.replace('(','')
            x = x.replace(')','')
            x = x.replace('@','')
            return(x)
    else:
        return 'asfimport'


def urls_handler(texts_comments):
    """
    Function to identify and replace links in issue's comments.
    :param text_comments: comments list
    :return texts_comments: comments texts without urls
    """
    def categorize_and_replace_links(text):
        for category, pattern in LINKS_PATTERNS.items():
            text = pattern.sub(f'[{category.upper()}_LINK]', text)
        return text

    texts_no_links = []
    for text in texts_comments:
        texts_no_links.append(categorize_and_replace_links(str(text)))

    split_url_comments_pattern = "([JIRA_ISSUE_COMMENT_LINK]\n\n"
    no_url, texts_comments = split_strings(texts_no_links, split_url_comments_pattern)
    texts_comments = [(a if a is not None else "") + (b if b is not None else "") for a, b in zip(no_url, texts_comments)]

    return texts_comments


def asfimport_handler_comments():
    """
    Function to extract useful information from the generated text of the bot asfimport only on the comments texts
    :return issues_df: the issues_df dataframe with the correct attribution and the cleaned text
    """
    asfimport_comments = issues_df[issues_df['user.login'] == 'asfimport']
    split_comments_pattern = "([migrated from JIRA]"
    infos_comments, texts_comments = split_strings(asfimport_comments['comment_body'], split_comments_pattern)

    texts_comments = urls_handler(texts_comments)
    asfimport_comments['info_asfimport'] = infos_comments
    asfimport_comments['user.login_extracted'] = asfimport_comments['info_asfimport'].apply(extract_pattern)
    asfimport_comments['comment_body'] = texts_comments
    asfimport_comments['user.login'] = asfimport_comments['user.login_extracted'].apply(lambda x: "asfimport" if x is None else x)
    asfimport_comments.drop(columns=['info_asfimport','user.login_extracted'],inplace=True)
    issues_df = pd.concat([issues_df[issues_df['user.login'] != 'asfimport'], asfimport_comments])    
    return issues_df


def asfimport_handler(issues_df):
    """
    Function to extract useful information from the generated text of the bot asfimport, 
        useful to save the majority of the issues texts imported from external platforms.
    :param issues_df: the issues_df dataframe with the texts of other developers assigned to 'asfimport'
    :return issues_df: the issues_df dataframe with the correct attribution and the cleaned text
    """
    asfimport_issues = issues_df[issues_df['issue_user'] == 'asfimport']
    separator = '\n\n\n\n---\nMigrated from'
    texts, infos = split_strings(asfimport_issues['issue_body'], separator)
    asfimport_issues['issue_body'] = texts
    asfimport_issues['info_asfimport'] = infos
    asfimport_issues['issue_user'] = asfimport_issues['info_asfimport'].apply(extract_pattern)
    asfimport_issues.drop(columns=['info_asfimport'],inplace=True)
    issues_df = pd.concat([issues_df[issues_df['issue_user'] != 'asfimport'], asfimport_issues])
    issues_df = asfimport_handler_comments()
    return issues_df


def issues_preprocessing():
    """
    Function to perform an initial filtering on noise elements in the issues data. 
    In particular, this procedure includes the noise of bots and urls in the issues texts.
    This function generates the final df 'issues_df_preprocessed.csv'
    """
    issues_df = pd.read_csv(os.path.join(PATH_INIT, 'issues_df.csv'))

    """
    After an exploratory analysis of the dataset, one of the issues with the users details was related to bot characterized as humans.
    Later the most frequent ones have been removed, and in a sub function have been extracted the relevant informations from the bot texts,
        used to import the issues from other platforms, in particular, 'asfimport'. 
    This function is related to the dataset in analysis, on other repositories, this details could be different.
    """
    issues_df = issues_df[issues_df['issue_user'] != 'dependabot[bot]']
    issues_df = issues_df[issues_df['issue_user'] != 'github-actions[bot]']
    issues_df = issues_df[issues_df['issue_user'] != 'Apache9']
    # The prior procedure has been applied also on the comments
    issues_df = issues_df[issues_df['user.login'] != 'Apache-HBase']
    issues_df = issues_df[issues_df['user.login'] != 'github-actions[bot]']

    # asfimport 
    issues_df = asfimport_handler(issues_df)
    issues_df.to_csv(os.path.join(PATH_INIT, 'issues_df_preprocessed.csv'), index=False)

def data_mining_function():
    print('JiT Reliability aggregation and APIs calls formatting...')
    jit_extraction()
    print('Commits retrieve...')
    retrieve_commits_info()
    print('Issues extraction...')
    issues_extraction()
    print('Issues aggregation...')
    issues_json_formatting()
    print('Issues preprocessing...')
    issues_preprocessing()


