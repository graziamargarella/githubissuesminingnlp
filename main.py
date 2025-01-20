from data_mining import data_mining_function
from topic_extraction import topic_model_step
from sentiment_analysis import sentiment_analysis_step
from feature_engineering import feature_engineering_step
from clustering import clustering_step


if __name__ == '__main__':
    data_mining = True
    topic_extraction = True
    sentiment_analysis = False #set to True to execute experiments
    feature_engineering = True
    clustering = True

    if data_mining:
        print('---- DATA MINING ----')
        data_mining_function()
    if topic_extraction:
        print('---- TOPIC EXTRACTION ----')
        topic_model_step() #mode='search' to execute experiments

    if sentiment_analysis:
        print('---- SENTIMENT ANALYSIS ----')
        sentiment_analysis_step()
        
    if feature_engineering:
        print('---- FEATURE ENGINEERING ----')
        feature_engineering_step() 

    if clustering:
        print('---- CLUSTERING USERS ----')
        clustering_step()  #mode='search' to execute experiments
    print('DONE')