import os
import pandas as pd
import matplotlib as plt
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, pipeline, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from datasets import load_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

PATH_DATA = './data/init'
PATH_RESULTS = './results/sentiment_analysis'
PATH_FINE_TUNING = './results/sentiment_analysis/roberta_finetuning'

BINARY_ROBERTA = 'siebert/sentiment-roberta-large-english'
TERNARY_ROBERTA = 'j-hartmann/sentiment-roberta-large-english-3-classes'

def sentiment_computing(comments, model=TERNARY_ROBERTA):
    """
    Function to compute the sentiment for given comments and model.
    :param comments: list of comments texts
    :param model: string of the model name on the huggingface platform
    :return results_df: dataframe of results with sentiment and sentiment score columns
    """
    tokenizer = AutoTokenizer.from_pretrained(model)
    sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer) #device=0 per cuda
    tokenized_comments = tokenizer(comments, truncation=True, padding=True, max_length=511)
    input_data = [{'input_ids': tokenized_comments['input_ids'][i],
                'attention_mask': tokenized_comments['attention_mask'][i]}
               for i in range(len(tokenized_comments['input_ids']))]
    
    results = []
    for item in tqdm(input_data):
        input_ids = item['input_ids']
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        with torch.no_grad():
            result = sentiment_pipeline(text[:1024])
        results.append(result)
    
    results_dicts = [item[0] for item in results]
    results_df = pd.DataFrame(results_dicts)
    return results_df

def dataframes_sentiment_enrichment(df, sentiment_df):
    """
    Function to add sentiment prediction to the original issues dataframes.
    :param df: dataframe of issues
    :param sentiment_df: dataframe of sentiment analysis results
    :return df: df dataframe with results.
    """
    df = df[~df['reactions.+1'].isna()]
    df['sentiment'] = sentiment_df['label']
    df['sa_score'] = sentiment_df['score']
    df['sentiment'].value_counts()
    return df


def roberta_fine_tuning():
    """
    Function to execute the finetuning process for the best model identified in the previous tests.
    This process has been applied on the 'github_gold.csv' labeled dataset of comments, and later validated on the 'issues_df_preprocessed.csv' dataframe.
    To perform this function with CUDA, the commented code execute the process on the available GPU.
    """
    df = pd.read_csv(os.path.join(PATH_FINE_TUNING, 'github_gold.csv'), sep=';')
    df.drop(['ID'], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df = df.rename(columns={'Polarity': 'label'})
    df['label'] = df['label'].apply(lambda x: 0 if x == 'negative' else 2 if x == 'positive' else 1)
    train_df, val_df = train_test_split(df, test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained(TERNARY_ROBERTA)
    model = RobertaForSequenceClassification.from_pretrained(TERNARY_ROBERTA)
    # model.to('cuda')

    training_datasets = Dataset.from_pandas(train_df)
    validation_datasets = Dataset.from_pandas(val_df)

    def tokenize_function(text):
        """
        Function that applies tokenization from a given text and returns the tokenized version.
        """
        tokenized = tokenizer(text['Text'], padding='max_length', truncation=True)
        return tokenized

    training_datasets = training_datasets.map(tokenize_function, batched=True)
    validation_datasets = validation_datasets.map(tokenize_function, batched=True)
    
    training_datasets = training_datasets.remove_columns(['Text','__index_level_0__'])
    validation_datasets = validation_datasets.remove_columns(['Text','__index_level_0__'])
    
    training_datasets.set_format('torch')
    validation_datasets.set_format('torch')

    def compute_metrics(p):
        """
        Utils function to compute evaluation metrics for classification: 
        :param p: prediction
        :return dict with accuracy, f1-score, precision and recall.
        """
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
    
    training_args = TrainingArguments(
        output_dir= PATH_FINE_TUNING,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_datasets,
        eval_dataset=validation_datasets,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.save_model(os.path.join(PATH_FINE_TUNING, "fine_tuned_sentiment_model"))

    execute_finetuning(val_df)
    comparison_finetuning()


def execute_finetuning(val_df):
    """
    Function that perform the evaluation of the finetuned model, given a validation set.
    """
    tokenizer = AutoTokenizer.from_pretrained(TERNARY_ROBERTA)
    sentiment_pipeline = pipeline("sentiment-analysis",model=os.path.join(PATH_FINE_TUNING, 'fine_tuned_sentiment_model'), tokenizer=tokenizer) #device=0 for GPU
    input = val_df['Text'].to_list()
    tokenized_input = tokenizer(input, truncation=True, padding=True, max_length=511)
    input_data = [{'input_ids': tokenized_input['input_ids'][i],
                    'attention_mask': tokenized_input['attention_mask'][i]}
                for i in range(len(tokenized_input['input_ids']))]
    results = []
    for item in tqdm(input_data):
        input_ids = item['input_ids']
        attention_mask = item['attention_mask']
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        with torch.no_grad():
            result = sentiment_pipeline(text[:1024])
        results.append(result)

    results_dicts = [item[0] for item in results]
    results_df = pd.DataFrame(results_dicts)
    results_df['Text'] = val_df['Text'].to_list()
    results_df['target'] = val_df['label'].to_list()

    results_df.to_csv(os.path.join(PATH_FINE_TUNING, 'sentiment_finetune.csv'), index=False)


def comparison_finetuning():
    """
    Function to print the evaluation metrics
    """
    df_no_finetune = pd.read_csv(os.path.join(PATH_FINE_TUNING, 'sentiment_no_finetune.csv'))
    df_finetune = pd.read_csv(os.path.join(PATH_FINE_TUNING, 'sentiment_finetune.csv'))
    df_no_finetune['label'] = df_no_finetune['label'].apply(lambda x: 0 if x == 'negative' else 2 if x == 'positive' else 1)
    
    labels = df_no_finetune['target'].to_list()
    preds = df_no_finetune['label'].to_list()

    def metrics_evaluation(labels, preds):
        """
        
        """
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        return acc,precision,recall,f1
    
    acc, precision, recall, f1 = metrics_evaluation(labels, preds)
    print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}')
    
    df_finetune['label'] = df_finetune['label'].apply(lambda x: 0 if x == 'negative' else 2 if x == 'positive' else 1)
    
    labels = df_finetune['target'].to_list()
    preds = df_finetune['label'].to_list()
    
    acc, precision, recall, f1 = metrics_evaluation(labels, preds)
    print(f'Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}')


def visualization():
    """
    Function to generate a plot for the distribution of the classes (negative, neutral and positive),
      across the different procedures. In particular, the 'Distr_Sentiment_Multi.pdf' image compares 
      the results of the sentiment analysis process applied to raw and preprocessed data and after the finetuning.    
    """
    df_zero_shot_raw = pd.read_csv(PATH_RESULTS + 'issues_raw_with_sentiment.csv')
    df_finetune = pd.read_csv(PATH_RESULTS + 'issues_preprocessed_with_sentiment_finetuned.csv')
    df_zero_shot_preprocessing = pd.read_csv(os.path.join(PATH_RESULTS, 'issues_preprocessed_with_sentiment.csv'))
    
    a = df_zero_shot_raw.groupby('sentiment')['sa_score'].count()
    b = df_zero_shot_preprocessing.groupby('sentiment')['sa_score'].count()
    c= df_finetune.groupby('sentiment')['sa_score'].count()
    df_multi = pd.concat([a, b, c], axis=1)

    x=np.arange(len(df_multi.index))
    x1 = x - 0.25
    x2 = x
    x3 = x + 0.25

    plt.bar(x1, a, width=0.25, label='Zero Shot Raw')
    plt.bar(x2, b, width=0.25, label='Zero Shot Preprocessing')
    plt.bar(x3, c, width=0.25, label='Fine Tuning')

    plt.xticks(x, df_multi.index)
    plt.legend()
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.savefig("Distr_Sentiment_Multi.pdf")
    plt.show()


def sentiment_analysis_step():
    """
    Pipeline step to execute all the sentiment analysis experiments to search the best configurations.
    """
    # Data Loading
    issues_df_raw = pd.read_csv(os.path.join(PATH_DATA, 'issues_df.csv'))
    comments_raw = issues_df_raw['comment_body'].dropna()
    comments_raw = comments_raw.tolist()

    # Raw test
    # Binary
    issues_df_binary_raw = dataframes_sentiment_enrichment(issues_df_raw, sentiment_computing(comments_raw, BINARY_ROBERTA))
    issues_df_binary_raw.to_csv(os.path.join(PATH_RESULTS, 'issues_raw_binary_with_sentiment.csv'))
    
    # Ternary
    issues_df_raw = dataframes_sentiment_enrichment(issues_df_raw, sentiment_computing(comments_raw, TERNARY_ROBERTA))
    issues_df_raw.to_csv(os.path.join(PATH_RESULTS, 'issues_raw_with_sentiment.csv'))

    # Preprocessed test
    issues_df_preprocessed = pd.read_csv(os.path.join(PATH_DATA, 'issues_df_preprocessed.csv'))
    comments = issues_df_preprocessed['comment_body'].dropna()
    comments = comments.tolist()
    
    # Binary
    issues_binary_df = dataframes_sentiment_enrichment(issues_df_preprocessed, sentiment_computing(comments, BINARY_ROBERTA))
    issues_binary_df.to_csv(os.path.join(PATH_RESULTS, 'issues_preprocessed_binary_with_sentiment.csv'))

    # Ternary
    issues_df = dataframes_sentiment_enrichment(issues_df_preprocessed, sentiment_computing(comments, TERNARY_ROBERTA))
    issues_df.to_csv(os.path.join(PATH_RESULTS, 'issues_preprocessed_with_sentiment.csv'))

    # Finetuning
    roberta_fine_tuning()

    visualization()

