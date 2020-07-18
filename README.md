# profiling-fake-news-spreaders
This repo contains code related to our submission for the task of Profiling Fake News Spreaders in PAN at CLEF 2020. 
The `software` directory include the scripts used in TIRA, verbatim. PyTorch model weights are required to run the scripts. Run these scripts the same as they would run in TIRA as:
```
./electra_{en/es}_{type}.py -i $inputDataset -o $outputDir
```
It expects data to be in XML format as specified by the organizers in the [task homepage](https://pan.webis.de/clef20/pan20-web/author-profiling.html). You will need to edit the shebang to match your environment.


The model weights and Ensemble Training Notebooks can be viewed/downloaded from Kaggle:
- [Notebook/Weights for English Dataset](https://www.kaggle.com/coseck/fork-of-electra-on-pan-fake-news-2b295d)
- [Notebook/Weights for Spanish Dataset](https://www.kaggle.com/coseck/spanish-electra-on-pan-fake-news)

EDA of the Dataset: [EDA Notebook](https://www.kaggle.com/coseck/pan2020-profiling-fake-news-spreaders-eda)


Training data can be requested from [Zenodo](https://zenodo.org/record/3692319#.XxG-gi0w1QI).

The desciption of the scripts are as follows:
- `electra_{en/es}_ensemble.py` : Runs the complete ensemble of 15 models on the given inputDataset

```
./electra_{en/es}__ensemble.py -i inputDatasetDir -o outputDir  -m savedModelsDir
```

- `electra_{en/es}_oneshot.py` : Runs the best model  found during once on the given inputDataset only once. 

```
./electra_{en/es}__oneshot.py -i inputDatasetDir -o outputDir  -m bestmodel.pt
```

- `electra_{en/es}_solo.py` : Creates the ensemble using 15 copies of the best model and runs it on the given inputDataset.
```
./electra_{en/es}__solo.py -i inputDatasetDir -o outputDir  -m bestmodel.pt
```

Requirements:
- This work reuses code written for another project which you can pull using the following command
    ```
    git clone https://github.com/cozek/trac2020_submission.git
    ```
- Other libraries:
    - PyTorch
    - Transformers
    - Pandas
    - Numpy
    - Scikit-learn


If code/paper was helpful, please cite:
```
TBD
```