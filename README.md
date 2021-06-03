# KRAM
A system for Question Answering using modern approaches in NLP and Knowledge Graphs build on top of [EmbedKGQA](https://github.com/malllabiisc/EmbedKGQA).

## Components

The project has the following main components:
1. Engine (All machine learning modules)
2. Flask server
3. UI

```
@inproceedings{saxena2020improving,
  title={Improving multi-hop question answering over knowledge graphs using knowledge base embeddings},
  author={Saxena, Apoorv and Tripathi, Aditay and Talukdar, Partha},
  booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
  pages={4498--4507},
  year={2020}
}
```

## Running the project

Before running the project:
- Clone the repository
- Go to the cloned folder: 
	```cd ./KRAM```
- Create a new virtual environment
	 ```python3 -m venv /path/to/new/virtual/environment```
- Activate the env:
	  ```source <venv>/bin/activate```
- Install the dependencies:
	  ```pip install -r requirements.txt```

Guidance for python venv, if you are not using linux https://docs.python.org/3/library/venv.html

There are two ways of running this project, descriibed below:
1. The training/test option
2. The UI option:
    - Start the flask server( KRAM/app/kram-server)
    - Start the frontend (KRAM/app/kram-frontend)


## Paramenters for training the model/engine:

```
--mode train --relation_dim 200 --hidden_dim 256 --gpu 1 --freeze 0 --batch_size 128 --validate_every 5 --hops 2 --lr 0.0005 --entdrop 0.1 --reldrop 0.2 --scoredrop 0.2 --decay 1.0 --model ComplEx --patience 5 --ls 0.0 --kg_type half --use_cuda 1 --gpu 0 --num_workers 0
```

Note: Set `--use_cuda 0` for CPU-only.

## Sample driver

```
engine = Engine()

print(engine.answer("which person directed the movies starred by Johnny Depp"))
```
