"""
Backend API client for DocuChat
"""
import requests
import json
from typing import Optional

BACKEND_URL = "http://localhost:8000"


def chat(user_input: str, data: str, session_id: Optional[str] = None) -> tuple[str, str]:
    """
    Send chat message to backend API and get response

    Args:
        user_input: User's question/input
        data: Document filename in storage
        session_id: Optional session ID for conversation context

    Returns:
        Tuple of (response answer, session_id)
    """
    url = f"{BACKEND_URL}/chat"

    payload = {
        "user_input": user_input,
        "data_source": data
    }

    if session_id is not None:
        payload["session_id"] = session_id

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        answer = result.get("response", {}).get("answer", "")
        session_id = result.get("session_id", "")
        return answer, session_id
    else:
        raise Exception(f"Error from backend: {response.status_code} - {response.text}")


def upload_file(file_path: str) -> str:
    """
    Upload document file to backend storage API

    Args:
        file_path: Local path to document file

    Returns:
        Storage path/URL of uploaded file
    """
    filename = file_path.split("\\")[-1].split("/")[-1]
    url = f"{BACKEND_URL}/uploadFile"

    with open(file_path, 'rb') as f:
        files = [('data_file', (filename, f, 'application/pdf'))]
        headers = {'accept': 'application/json'}

        response = requests.post(url, headers=headers, files=files)

    if response.status_code == 200:
        return response.json().get("file_path", "")
    else:
        raise Exception(f"Error uploading file: {response.status_code} - {response.text}")
