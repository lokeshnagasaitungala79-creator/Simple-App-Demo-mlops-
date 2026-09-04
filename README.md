# Simple APP MLOps

A simple Machine Learning project for learning and implementing **MLOps practices** using Python, Conda, Git, DVC, and MLflow.

---

## 1. Create Project Directory

```powershell
mkdir "Simple APP Mlops"
cd "Simple APP Mlops"
```

---

## 2. Create Conda Environment

Create a Conda environment with Python 3.13:

```powershell
conda create -n wineq python=3.13 -y
```

Activate the environment:

```powershell
conda activate wineq
```

Check Python version:

```powershell
python --version
```

---

## 3. Create Requirements File

In Windows PowerShell, use:

```powershell
New-Item requirements.txt -ItemType File
```

> `touch requirements.txt` is a Linux/macOS command and does not work by default in PowerShell.

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 4. Create README File

Create the README file using PowerShell:

```powershell
New-Item README.md -ItemType File
```

Open it in VS Code:

```powershell
code README.md
```

---

## 5. Install Machine Learning Packages

Add the following packages to `requirements.txt`:

```text
dvc
scikit-learn
pandas
numpy
matplotlib
mlflow
```

Install them:

```powershell
pip install -r requirements.txt
```

### Important

Use:

```text
scikit-learn
```

in `requirements.txt`, not:

```text
sklearn
```

However, in Python the import is:

```python
import sklearn
```

---

## 6. Verify Installed Packages

Check DVC:

```powershell
dvc --version
```

Check Python:

```powershell
python --version
```

Check Scikit-learn:

```powershell
python -c "import sklearn; print(sklearn.__version__)"
```

Check MLflow:

```powershell
mlflow --version
```

---

## 7. Initialize Git

Initialize a Git repository:

```powershell
git init
```

Check Git status:

```powershell
git status
```

Add all files:

```powershell
git add .
```

Create the first commit:

```powershell
git commit -m "Initial commit"
```

---

## 8. Initialize DVC

Initialize DVC inside the Git repository:

```powershell
dvc init
```

Check DVC status:

```powershell
dvc status
```

---

## 9. Add Dataset Using DVC

Place the dataset inside the project.

Example:

```text
data/
└── raw/
    └── dataset.csv
```

Add the dataset to DVC:

```powershell
dvc add data/raw/dataset.csv
```

Check DVC status:

```powershell
dvc status
```

Add the generated DVC file to Git:

```powershell
git add data/raw/dataset.csv.dvc .gitignore
```

Commit:

```powershell
git commit -m "Track dataset with DVC"
```

---

## 10. Git History

View Git commit history:

```powershell
git log
```

View a short version:

```powershell
git log --oneline
```

---

## 11. PowerShell History

View commands executed in the current PowerShell session:

```powershell
history
```

Show the last 10 commands:

```powershell
Get-History -Count 10
```

---

## 12. Useful PowerShell Commands

Create a file:

```powershell
New-Item filename.txt -ItemType File
```

Create a folder:

```powershell
mkdir folder_name
```

List files:

```powershell
dir
```

Display file contents:

```powershell
Get-Content filename.txt
```

Open VS Code:

```powershell
code .
```

Show current directory:

```powershell
Get-Location
```

---

## 13. MLOps Workflow

```text
Data
 ↓
Data Versioning (DVC)
 ↓
Data Preparation
 ↓
Feature Engineering
 ↓
Model Training
 ↓
Model Evaluation
 ↓
Experiment Tracking (MLflow)
 ↓
Model Versioning
 ↓
Deployment
 ↓
Monitoring
```

---

## 14. Technologies

* Python
* Conda
* Pandas
* NumPy
* Scikit-learn
* Git
* GitHub
* DVC
* MLflow
* VS Code


tox command .
```bash
tox 
```
for rebuilding .
``` bash
tox -r

pytest command
```bash
pytest -v

setup commands : -
pip install -e.

build your own package command:-
python setup.py sdist bdist_wheel

---


python -m mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./artifacts \
  --host 0.0.0.0 \
  --port 1234

## Author

**Lokesh Naga Sai**

CSE-AIML Student
