import os
import numpy as np
import regex as re
import pandas as pd
from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from bertopic.vectorizers import ClassTfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from bertopic import BERTopic
from octis.evaluation_metrics.coherence_metrics import Coherence
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from sentence_transformers import SentenceTransformer
import plotly.io as pio
import kaleido


nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
pio.kaleido.scope.default_format = 'pdf'


PATH_DATA = './data/init'
PATH_RESULTS = './results/topic_extraction'
EXPERIMENTS_EXECUTIONS = 5


class BERTopicModelImpl(AbstractModel) :
  """
  Class to encapsulate the BERTopic implementation for octis metrics evaluation process.
  """
  def __init__(self, min_cluster_size=None,
               min_sample_size=None,
               embeddings=None,
               nr_topics=None,
               outlier_threshold=0,
               bm25_weigthing=False,
               reduce_frequent_words=False) :

    super().__init__()
    self.hyperparameters = dict()
    self.hyperparameters['min_cluster_size'] = min_cluster_size
    self.hyperparameters['min_sample_size'] = min_sample_size
    self.hyperparameters['embeddings'] = embeddings
    self.BERTopic_model = None
    self.BERTopic_topics = None
    self.outlier_threshold = outlier_threshold
    self.bm25 = bm25_weigthing
    self.reduce_words = reduce_frequent_words
    self.ctfidf_model = ClassTfidfTransformer(bm25_weighting=self.bm25, reduce_frequent_words=self.reduce_words)

    vectorizer_model = CountVectorizer(ngram_range=(1, 2), min_df=10)

    self.init_params = {'vectorizer_model' : vectorizer_model, 'ctfidf_model': self.ctfidf_model}

    if self.hyperparameters['min_cluster_size'] is not None :
      hdbscan_model = HDBSCAN(metric='euclidean',
                              cluster_selection_method='eom',
                            prediction_data=False,
                              min_cluster_size=self.hyperparameters['min_cluster_size'],
                              min_samples=self.hyperparameters['min_sample_size'])
      self.init_params['hdbscan_model'] = hdbscan_model

    if nr_topics is not None :
      self.init_params['nr_topics'] = nr_topics

    self.BERTopic_model = BERTopic(**self.init_params)

  def train_model(self, dataset):
    bertdata = dataset.get_corpus()
    self.BERTopic_topics, _ = self.BERTopic_model.fit_transform(bertdata,
                                                           embeddings=self.hyperparameters['embeddings'])

    self.BERTopic_topics = self.BERTopic_model.reduce_outliers(bertdata, self.BERTopic_topics, strategy="c-tf-idf", threshold=self.outlier_threshold)

    bertopic_topics = [
        [topicwords[0] for topicwords in self.BERTopic_model.get_topic(i)[:10]]
          for i in range(len(set(self.BERTopic_topics)) - 1)]

    result = dict()
    result['topics'] = bertopic_topics
    return result


def dataset_definition():
    """
    Function to define the octis Datasets with the joined field of the title and bodies of the issues, both raw and preprocessed.
    :return dataset_raw: octis Dataset
    :return docs: Dataset corpus
    :return dataset_preprocessed: octis Dataset
    :return preprocessed_docs: Dataset corpus 
    """
    issues_df = pd.read_csv(os.path.join(PATH_DATA, 'issues_df_preprocessed.csv'))
    issues_df = issues_df[['issue.names','num_comments', 'issue_title', 'issue_body', 'issue_user', 'issue_updated_at', 
                        'issue_labels_names', 'issue_author_association', 'issue_created_at', 'issue_updated_at.1', 'issue_closed_at', 
                        'is_pull_request']].drop_duplicates()

    titles = [str(item) for item in issues_df['issue_title']]
    bodys = [str(item) for item in issues_df['issue_body']]
    issues_df['raw_text'] = [' '.join([title, body]) for title, body in zip(titles, bodys)]

    documents = list(issues_df['raw_text'])

    dataset_raw = Dataset(documents)
    docs = dataset_raw.get_corpus()

    preprocessed_documents = [preprocessing_pipeline(doc) for doc in docs]
    dataset_preprocessed = Dataset(preprocessed_documents)
    preprocessed_docs = dataset_preprocessed.get_corpus()
    
    return dataset_raw, docs, dataset_preprocessed, preprocessed_docs


def preprocessing_pipeline(text, min_token_length=2, max_token_length=30, min_token_freq=0.007, min_tokens_per_doc=1):
    """
    Proposed preprocessing pipeline to clean the issues texts.
    :param text: the input string to process.
    :param min_token_length: minimum length of the tokens, default = 2.
    :param max_token_length: maximum lenght of the tokens, default = 30.
    :param min_token_freq: minumum frequency of the tokens, default = 0.007.
    :param min_tokens_per_doc: minumum number of tokens per document, default = 1. 
    :return a string of the remaining tokens of the input text. 
    """
    jira_id_pattern = re.compile(r'(camel|hadoop|hbase|ivy|jcr|log4j2|lucene|mahout|openjpa|velocity|xercesc|amq)-\d{1,5}')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if not re.match(jira_id_pattern, token)]
    tokens = [token for token in tokens if not re.match(r'http\S+', token)]
    tokens = [re.sub(r'[^\w\s\'\-]', '', token) for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if min_token_length <= len(token) <= max_token_length]

    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    if total_tokens > 0:
        token_freq = {token: count / total_tokens for token, count in token_counts.items()}
        tokens = [token for token in tokens if token_freq.get(token, 0) >= min_token_freq]

    if len(tokens) < min_tokens_per_doc:
        return None
    return ' '.join(tokens)


def metrics_calculation(results):
    """
    Evaluation metrics from octis library calculation.
    :param results: results dictionary of a BERTopicModelImpl object
    :return num_topics: number of topics identified in the results
    :return cv: Topic Coherence, C_V measure
    :return npmi: Topic Coherence, C_NPMI measure 
    :return td: Topic Difference
    """
    min_topic_length = 3
    cleaned_topics = [[token for token in topic if token and '\n' not in token]
                    for topic in results['topics']]
    cleaned_topics = [topic for topic in cleaned_topics if len(topic) > min_topic_length]

    token2id = {token: id for id, token in enumerate(set(token for topic in cleaned_topics for token in topic))}
    cleaned_topics_ids = [[token2id[token] for token in topic] for topic in cleaned_topics]

    num_topics = len(cleaned_topics_ids)
    cv = Coherence(measure='c_v',topk=3).score({'topics': cleaned_topics_ids})
    npmi = Coherence(measure='c_npmi',topk=3).score({'topics': cleaned_topics_ids})
    td = TopicDiversity(topk=3).score({'topics': cleaned_topics_ids})

    print(f'Num Topics: {num_topics}, CV: {cv}, NPMI: {npmi}, TD: {td}')
    return num_topics, cv, npmi, td


def embedding_models_calculation(docs, preprocessed_docs):
    """
    Function to retrieve embeddings models
    :param docs: corpus of raw issues texts
    :param preprocessed_docs: corpus of preprocessed issues texts
    :return embedding_1: the vector representation given by the all-MiniLM-L6-v2 on preprocessed data
    :return embedding_2: the vector representation given by the all-MiniLM-L6-v2 on raw data
    :return embedding_3: the vector representation given by the all-mpnet-base-v2 on preprocessed data
    :return embedding_4: the vector representation given by the all-mpnet-base-v2 on raw data
    """
    embedding_model_mini = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model_mpnet = SentenceTransformer('all-mpnet-base-v2')

    embeddings_1 = embedding_model_mini.encode(preprocessed_docs, show_progress_bar=True) #mini + preprocess
    np.save(os.path.join(PATH_RESULTS,'embeddings_1.npy'), embeddings_1)
    embeddings_2 = embedding_model_mini.encode(docs, show_progress_bar=True) # mini
    np.save(os.path.join(PATH_RESULTS,'embeddings_2.npy'), embeddings_2)
    embeddings_3 = embedding_model_mpnet.encode(preprocessed_docs, show_progress_bar=True) #mpnet + preprocess
    np.save(os.path.join(PATH_RESULTS,'embeddings_3.npy'), embeddings_3)
    embeddings_4 = embedding_model_mpnet.encode(docs, show_progress_bar=True) #mpnet
    np.save(os.path.join(PATH_RESULTS,'embeddings_4.npy'), embeddings_4)

    return embeddings_1, embeddings_2, embeddings_3, embeddings_4


def best_embedding_loading():
   """
   Function to load the best embedding algorithm, pre-computed
   :return embeddings: the vector representation given by the all-MiniLM-L6-v2 on preprocessed data
   """
   embeddings = np.load(os.path.join(PATH_RESULTS,'embeddings_1.npy'))
   return embeddings


def benchmark_function(config, name='embedding', best_config=None):
  """
  Function to perform evaluations on the different hyperparameters of the BERTopic model configuration,
    in order to find the best hyperparameters for the dataset analysed.
  :param config: a dict containing the parameters to test.
  :param name: a string that could be ['embedding', 'outliers','term-weighting'] to save the results and perform a different configuration
  :param best_config: a dict containing the best parameters find before, useful to reduce the space of the experiments.
  :return df: the results found in the experiments performed. 
  """ 
  results = []
  bertopic_model = None
  for c in config:
    for _ in range(EXPERIMENTS_EXECUTIONS):
      if name == 'embedding':
        bertopic_model = BERTopicModelImpl(embeddings=c['embedding'])
      elif name == 'outliers':
        bertopic_model = BERTopicModelImpl(embeddings=best_config['embedding'], outlier_threshold=c['outliers'])
      elif name == 'term-weighting':
        bertopic_model = BERTopicModelImpl(embeddings=best_config['embedding'], outlier_threshold=best_config['outliers'], reduce_frequent_words=c['reduce_frequent_words'], bm25_weigthing=c['bm25_weigthing'])
      else:
        return -1
      model_results = bertopic_model.train_model(c['dataset'])
      num_topics, cv, npmi, td = metrics_calculation(model_results)
      result = {**c, 'num_topics': num_topics, 'cv': cv, 'npmi': npmi, 'td': td}
      results.append(result)
  
  df = pd.DataFrame(results)
  return df


def execution(config):
  """
  Function to adapt the best configuration found to the representation of the issues dataset.
  In particular, from the approximative 120 topics, analysing the hierarchical clustering tree, have been set as optimal, 11 topics.
  :param config: the best configuration found in previous analysis
  :return bertopic_model: model instance for further analysis
  :return results: results computed
  """
  bertopic_model = BERTopicModelImpl(embeddings=config['embedding'], outlier_threshold=config['outliers'], 
                                    reduce_frequent_words=config['reduce_frequent_words'], bm25_weigthing=config['bm25_weigthing'],
                                    nr_topics=11)
  results = bertopic_model.train_model(config['dataset'])
  metrics_calculation(results)
  return bertopic_model, results


def visualizations(bertopic_model):
  """
  Function to plot top word for each topic using a barchart and the similarity matrix of the topics,
    gives as outputs two pdf files of the plots.
  :param bertopic_model: model to use for the visualizations.
  """
  bertopic_model.visualize_barchart(topics=range(10))\
    .update_layout(font=dict(size=20),width=1265, height=700, margin=(dict(t=70,b=0,l=10,r=10)),\
                   title=(dict(y=0.98, font=dict(size=26))))\
                  .write_image(os.path.join(PATH_RESULTS,'top_words_label.pdf'))
  bertopic_model.visualize_heatmap()\
    .update_layout(font=dict(size=20),width=1000, height=700, margin=(dict(t=50,b=0,l=10,r=10)),\
                  title=(dict(y=0.98, font=dict(size=26))))\
                  .write_image(os.path.join(PATH_RESULTS,'similarity_label.pdf'))


def topics_dataset_formatting(model, docs, embeddings):
   """
   Computing and results of the model and aggregation of the original dataset with the topic for each issue.
   :param model: BERTopic model already setted
   :param docs: final documents to map
   :param embeddings: embedded representation of the docs
   :return issues_df: updated
   """
   issues_df = pd.read_csv(os.path.join(PATH_DATA, 'issues_df_preprocessed.csv'))
   issues_df = issues_df[['issue.names','num_comments', 'issue_title', 'issue_body', 'issue_user', 'issue_updated_at', 
                        'issue_labels_names', 'issue_author_association', 'issue_created_at', 'issue_updated_at.1', 'issue_closed_at', 
                        'is_pull_request']].drop_duplicates()
   topics, probs = model.transform(documents=docs, embeddings=embeddings)
   issues_df['topic'] = topics
   issues_df['topic_prob'] = probs
   return issues_df


def topics_experiments_execution():
  """
  Function describing and executing all the configurations for topics experiments to find the best hyperparameter configuration.
  :return model: dict of the results obtained by the best model
  """
  dataset_raw, docs , dataset_preprocessed, preprocessed_docs = dataset_definition()
  embeddings_1, embeddings_2, embeddings_3, embeddings_4 = embedding_models_calculation(docs, preprocessed_docs)
  
  embeddings_setup = [
      {'embedding': embeddings_1, 'dataset': dataset_preprocessed},
      {'embedding': embeddings_2, 'dataset': dataset_raw},
      {'embedding': embeddings_3, 'dataset': dataset_preprocessed},
      {'embedding': embeddings_4, 'dataset': dataset_raw}
    ]

  representation_df = benchmark_function(embeddings_setup)
  representation_df.to_csv(os.path.join(PATH_RESULTS, 'representation_experiment.csv')) 

  best_config = {
      'dataset': dataset_preprocessed,
      'embedding': embeddings_1,
      'outliers': 0,
      'reduce_frequent_words': False, 
      'bm25_weigthing': False
    }
  
  outliers_parameters = [
  {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.01}, 
  {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.1}, 
  {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.2}, 
  {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.3}]

  outliers_df = benchmark_function(outliers_parameters, 'outliers', best_config)
  outliers_df.to_csv(os.path.join(PATH_RESULTS, 'outliers_experiment.csv'))

  best_config['outliers'] = 0.1

  term_weighting = [
      {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.1, 'reduce_frequent_words': False, 'bm25_weigthing': False},
      {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.1, 'reduce_frequent_words': False, 'bm25_weigthing': True},
      {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.1, 'reduce_frequent_words': True, 'bm25_weigthing': False},
      {'embedding': embeddings_1, 'dataset': dataset_preprocessed, 'outliers': 0.1, 'reduce_frequent_words': True, 'bm25_weigthing': True}
    ]

  term_weighting_df = benchmark_function(term_weighting, 'term_weighting', best_config)
  term_weighting_df.to_csv(os.path.join(PATH_RESULTS, 'term_weighting_experiment.csv'))
  best_config['reduce_frequent_words'] = True

  model, results = execution(best_config)
  model.BERTopic_model.save(os.path.join(PATH_RESULTS, "bertopic_model"), serialization="pickle")
  return model


def topic_model_step(mode='load'):
  """
  Function to call to execute the topic model in the final pipeline.
  :param mode: string of the modality of execution, if to 'load' an existing model or 'search' for the best one.
  :return issues_df: dataset of the preprocessed issue with the topic assegnation.
  """
  _, docs, _, preprocessed_docs = dataset_definition()
  if mode == 'load':
    topic_model = BERTopic.load(os.path.join(PATH_RESULTS, "bertopic_model"))
  elif mode == 'search':
    topic_model = topics_experiments_execution()
  else:
     return -1

  visualizations(topic_model)
  issues_df = topics_dataset_formatting(topic_model, docs, embeddings=best_embedding_loading())
  return issues_df
