"""Inference microservice for the titer-prediction model.

A thin FastAPI layer over the trained models. The API converts a single
experiment (OpenAPI ``/predict`` payload) into a one-row-per-timestep DataFrame
and runs it through the *same* preprocessing + model used in training — the API
never re-implements model logic. Both trained models are supported; the one to
serve is chosen by ``MODEL_PATH`` (``.joblib`` → XGBoost baseline, ``.eqx`` →
neural CDE).
"""
