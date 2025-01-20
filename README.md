# GitHub Issue Mining to enhance Developer Profiling: A pipeline based on NLP tasks

In order to reconstruct the experiments of the pipeline are necessary this actions:
1. Install the required packages from the `requirements.txt` file in a conda environment.
2. Extract in the folder 'data/JiTReliability' the dataset available at https://github.com/lining-nwpu/JiTReliability, before from the zip files and later from the folders. 
3. Generate from the personal GitHub account a token and replace with this personal string the variable `GIT_TOKEN` in the `utils.py` file.
4. Select from the main script which step of the pipeline to execute (setting all `True` will execute all the experiments).
    For the topic extraction or the clustering experiments it is possible to load the best model found in the experiments.