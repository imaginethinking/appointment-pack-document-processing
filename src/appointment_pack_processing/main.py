"""Uvicorn entry point for the document-processing API."""

from appointment_pack_processing.app import create_app

app = create_app()