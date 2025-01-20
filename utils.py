import requests
import time

# Change the token with the generated one
GIT_TOKEN = ''


# JiT Reliability Repositories Names Mapping from GitHub to GH API calls
REPOS_MAPPING = {
    'ActiveMQ': 'apache/activemq',
    'Ant': 'apache/ant',
    'Camel': 'apache/camel',
    'Derby': 'apache/derby',
    'Geronimo': 'apache/geronimo',
    'Hadoop': 'apache/hadoop',
    'HBase': 'apache/hbase',
    'IVY': 'apache/ant-ivy',
    'JCR': 'apache/jackrabbit',
    'JMeter': 'apache/jmeter',
    'LOG4J2': 'apache/logging-log4j2',
    'LUCENE': 'apache/lucene',
    'Mahout': 'apache/mahout',
    'OpenJPA': 'apache/openjpa',
    'Pig': 'apache/pig',
    'POI': 'apache/poi',
    'VELOCITY': 'apache/velocity-engine',
    'XERCESC': 'apache/xerces-c'
}

"""
JiT Reliability files terminations:
    - **_commitsID.csv = software changes index;
    - **_failures.csv = issues names and date;
    - **_input.csv = software changes, column "bug" indicates if a software change contains software defects;
    - **_input_EB.csv = software changes, column "bug" indicates if a software change contains early exposed defects;
    - **_mapping.csv = software change-defect mapping.
"""
JIT_EXTENSIONS = [
    'commitsID.csv',
    'input.csv',
    'input_EB.csv',
    'failures.csv'
]

def fetch_gh_data(call, auth_token=GIT_TOKEN):
    """
    Fetch events details from GitHub API in JSON format.
    :param call: GitHub API link
    :param auth_token: GitHub authorization token.
    :return: Events details in JSON format
    """
    headers = {'Authorization': 'Token ' + auth_token} if auth_token else None
    response = requests.get(call, headers=headers)
    if response.status_code == 200:
        data = response.json()
    elif response.status_code == 403:
        if int(response.headers['X-RateLimit-Remaining']) == 0:
            # Sleep until reset
            target_timestamp = int(response.headers['X-RateLimit-Reset'])
            current_timestamp = int(time.time())  # Get the current Unix timestamp
            seconds_to_sleep = target_timestamp - current_timestamp  # Calculate the time difference
            print(
                f'Rate limit of {response.headers["X-RateLimit-Limit"]} reached, sleeping until {target_timestamp + 1}')
            if seconds_to_sleep > 0:
                time.sleep(seconds_to_sleep + 1)
            return fetch_gh_data(call, auth_token)
        else:
            print(response.text)
            exit(1)
    else:
        data = f'NOT FOUND'
    return data