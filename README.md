# from Notebooks 2 MLOps


Data Science task (notebooks2MLOps) - phase 1 

- this project have three important phases with three repos to represent how i go throw all the project and how i handle any data problem from notebooks creating to Developing a Model Inference API 

1- it's first taks about notebooks and how we deal with any data problem after i finish `00_allRoadslead2BrainStorming.ipynb` it's the notebook have all the ideas about how i think to made all of this and specially the markdown text i write after each step 

2- After i finish it i start split it into clearly reading strucutre notebook but i still prefer to put it 

3- the next step in the task i made is converting this notebook to best foundation practices MLOps could be part of cloud_deployment (actully not fully in this phase)

4- try to read `README.md` file to start project after cloning it (don't forget to add `data`) folder have original data of `train_data.csv` , `train_labels.csv` and `test_data.csv`


#### First pahse i called it `Model-to-App Packaging Production Codebase Design`
- with following this steps i made:

1 - **From Jupyter to Application Code Transformation**

2 - **Python Dependency Management Setup with Poetry**

3 - **Python Parametrization Setup with Pydantic**

4 - **Python Logging Setup with Loguru**

5 - **Foundations of Codebase Architecture & Clean Code Techniques for Software Excellence** (try my best)

6 - **Streamlining Code Quality Linters and Formatters**

7 - **Code Automation with Makefiles**

8 - **CICD with GitHub Actions**

#### Second Phase i called it `From Monolith to Microservices architecture`

in machine learning projects, you always have two main phases: first, you train your model, and then you use this trained model to make predictions (inference). it's important to keep these two steps separate — training and inference should not be mixed together in the same code or service.

so, to make things clean and production-ready, i split the project into two microservices: one for building (training) the model, and another for running inference (making predictions). this way, each part does its own job, and you can update or scale them independently.

the idea is to move from a single big (monolithic) app to a microservices setup, where you have a "model builder" microservice and a "model inference" microservice. this separation makes the system more robust and easier to manage.


### Last Phase i called it `Model Inference API`

- it's the same with MLops task assignment but depend on the second phase without training model again or we just call `model_infernce_service` also i made it before in my previous project: [Rental Price Prediction App to Micro Transformation API Development](https://github.com/Mohammed-abdulaziz-eisa/Rental-Price-Prediction-App-to-Micro-Transformation-API-Development). 

- but in this assisment i prefer to use the same mlops task structure to not confuse the reviwer of the task 

##  Start 

To get started with the WithSecure ML Pipeline, follow these steps:

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mohammed-abdulaziz-eisa/withsecure-phase1
   cd withsecure-phase1
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Run the complete pipeline**
   ```bash
   make runner
   ```

##  Important Notes

- Ensure that a `data` folder exists in the project root directory have the original data of the project before running the pipeline.

#### a bad part for `data` :

- which i just load the data from the path data but if the data have headers i will start to make Database Setup and Python Connectivity with SqlAlchemy 

why it's bad approach ?
While this approach is convenient for managing data during development it’s not suitable for production scenarios The reason is that real world datasets can be enormous and three major problems specially with if we work with or storing terabytes of data i will not talk more about this but i implement it before in a project i will attach a link to see how i implement it (actually to represent myself)

For a example of robust data management and database integration in an MLOps context, please go to my previous project: [Rental Price Prediction End-to-End MLOps Project](https://github.com/Mohammed-abdulaziz-eisa/Rental-Price-Prediction-End-to-End-MLOps-Project). In particular, the `src/db` directory you will see that how  database interaction layer with SQLAlchemy ORM models  and SQLite database 
i use DBeaver to mange it and crating engine  

##  System Architecture

The ML system follows a clean, layered architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                       │
├─────────────────────────────────────────────────────────────┤
│  runner.py - Main application entry point                 │
│  ModelService - High-level model management               │
├─────────────────────────────────────────────────────────────┤
│                    Pipeline Layer                          │
├─────────────────────────────────────────────────────────────┤
│  collection.py - Data loading and validation              │
│  preparation.py - Feature preprocessing (SMOTE + Scaling)  │
│  model.py - Model training and validation                 │
├─────────────────────────────────────────────────────────────┤
│                    Configuration Layer                     │
├─────────────────────────────────────────────────────────────┤
│  config.py - Centralized configuration management          │
│  Environment variables and fallback handling               │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Layer                    │
├─────────────────────────────────────────────────────────────┤
│  Poetry for dependency management                          │
│  Make for automation                                       │
│  GitHub Actions for CI/CD                                  │
└─────────────────────────────────────────────────────────────┘
```

## Development

### Code Quality
```bash
# Format and lint
make format
make lint

# Type checking
make type-check

# Run all checks
make check
```

### Testing
```bash
# Run tests
make test

# Install dependencies
make install

# Clean up
make clean
```

### Training and Inference
```bash
# Train model
make train

# Run inference
make predict

# Complete workflow
make runner
```

## Configuration

The pipeline uses centralized configuration with environment variable support:

```bash
# Required environment variables
export TRAIN_DATA_PATH=./data/train_data.csv
export TRAIN_LABELS_PATH=./data/train_labels.csv
export TEST_DATA_PATH=./data/test_data.csv
export OUTPUT_PATH=./data/test_labels.csv
export MODEL_PATH=./src/model/models
export MODEL_NAME=logistic_regression_v1

# Optional
export LOG_LEVEL=INFO
export RANDOM_STATE=42
```

## Model Details

- **Algorithm**: Logistic Regression with L2 regularization
- **Preprocessing**: SMOTE for imbalance + RobustScaler
- **Validation**: 5-fold stratified cross-validation
- **Performance**: Handles 10,000 features efficiently
- **Output**: Binary predictions (-1, 1)


## Project Structure

```
withsecure/
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── model/             # ML pipeline and services
│   └── runner.py          # Main application
├── docs/                  # documentation
├── tests/                 # Test suite
├── .github/               # CI/CD workflows
├── pyproject.toml         # Poetry dependencies
├── Makefile               # Automation scripts
└── README.md              # This file
```


### Debug Mode
```bash
export LOG_LEVEL=DEBUG
poetry run python src/runner.py
```


---

** For detailed documentation, guides, and examples, see [docs/README.md](docs/README.md)**