## 0.8.0 (2023-01-29)

### Feat

- **frame_manager_base**: added a new on_ends_stream event

## 0.7.1 (2023-01-29)

### Fix

- **frame_yolo_v7.py**: always returned the original frame instead the processed_frame

## 0.7.0 (2023-01-29)

### Feat

- **aipose**: add support to work with videos and refactor api

### Refactor

- **frame-and-stream**: apply isort and fixed __all__

## 0.6.0 (2023-01-29)

### Feat

- **aipose**: add support to work with videos and refactor api

## 0.5.1 (2023-01-28)

### Refactor

- **aipose**: run isort over the code

## 0.5.0 (2023-01-28)

### Feat

- **frame_managers.py**: added new managers

### Fix

- **model.py**: added int type to x, y and conf

### Refactor

- **webcam.md**: update webcam.md code and style

## 0.4.0 (2023-01-27)

### Feat

- **webcam**: added managers to add logic to webcam

## 0.3.1 (2023-01-27)

### Feat

- **model.py**: currently the yolo v7 pose model is downloaded into home/.aipose folder
- **model.py**: the model is downloaded into home folder if doesn't exists

### Fix

- **pyproject.toml**: update outdated reference to cli execution

## 0.1.1 (2023-01-24)

## 0.2.1 (2023-01-24)

### Feat

- **package**: set up build package with a function and added poe the poet to add scripts
- init project

### Fix

- **pyproject.toml**: select packages to install in pyproject.toml
