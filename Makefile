install:
	pip install -r requirements.txt

data:
	PYTHONPATH=src python -m flower_pipeline.generate_data

pipeline:
	PYTHONPATH=src python -m flower_pipeline.pipeline

train:
	PYTHONPATH=src python -m flower_pipeline.train

test:
	PYTHONPATH=src pytest -q

run:
	PYTHONPATH=src uvicorn app.main:app --reload
