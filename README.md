# Celsius to Fahrenheit ML Web App

This repository deploys the **final TensorFlow model** from `celsius_to_fahrenheit.ipynb`.

Final notebook model:

- Input training data: Celsius/Fahrenheit pairs from the notebook
- Architecture: `Dense(4) -> Dense(4) -> Dense(1)`
- Loss: `mean_squared_error`
- Optimizer: `Adam(0.1)`
- Epochs: `800`

## Files required for GitHub / Render

- `app.py`
- `requirements.txt`
- `.python-version`
- `render.yaml`
- `templates/index.html`
- `static/style.css`

The notebook is included only as the original reference/training notebook. Render runs `app.py`.

## Render

Create a Web Service from the GitHub repository. If Render reads `render.yaml`, no manual commands are needed.

Manual settings, if required:

Build command:

    pip install -r requirements.txt

Start command:

    gunicorn app:app --workers 1 --timeout 120
