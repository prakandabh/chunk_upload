"""
HuggingFace Audio Uploader

This script downloads audio files from a HuggingFace repository and uploads them
to an API with metadata.

TRACKING MECHANISM:
It stores upload tracking data in a JSON file within the HuggingFace repository itself. This allows multiple team
members running the script from different devices to share the same upload history.

Main workflow:
1. Load configuration from environment variables
2. Authenticate with API and load template
3. Download metadata AND tracking file from HuggingFace
4. Process audio files in batches:
   - Download main audio and chunks from HuggingFace
   - Prepare metadata according to template
   - Upload to API
   - Update tracking file in HuggingFace repo
   - Clean up downloaded files
"""

import csv
import json
import os
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Callable
from huggingface_hub import HfApi, hf_hub_download
import httpx
from dotenv import load_dotenv

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
# All configurable values are defined here for easy modification.
# This includes environment variable names, default values, file paths,
# tracking file structure, API endpoints, and all string literals used in the application.

# --------------------------------------------------------------------------
# Environment Variables Keys
# --------------------------------------------------------------------------
# These are the names of environment variables that will be read from .env file

ENV_HF_REPO_ID = "HF_REPO_ID"              # HuggingFace repository ID
ENV_HF_TOKEN = "HF_TOKEN"                  # HuggingFace API token (REQUIRED for writing tracking file)
ENV_API_BASE_URL = "API_BASE_URL"          # Base URL of the upload API
ENV_API_USERNAME = "API_USERNAME"          # Username for API authentication
ENV_API_PASSWORD = "API_PASSWORD"          # Password for API authentication
ENV_TEMPLATE_NAME = "TEMPLATE_NAME"        # Name of metadata template to use
ENV_TEMPLATE_ID = "TEMPLATE_ID"            # ID of metadata template (alternative to name)
ENV_BATCH_SIZE = "BATCH_SIZE"              # Number of files to process in each batch
ENV_BATCH_DELAY_SECONDS = "BATCH_DELAY_SECONDS"  # Delay between batches to avoid rate limiting
ENV_HF_SKIP_REMOTE_FILE_LIST = "HF_SKIP_REMOTE_FILE_LIST"  # Skip fetching HF file list (for large repos)


# --------------------------------------------------------------------------
# Directory and File Names
# --------------------------------------------------------------------------
# File system paths and names used by the application

DIR_WORKDIR = "workdir"                    # Working directory for temporary file downloads
DIR_AUDIO = "audio"                        # Subdirectory containing audio files in HF repo
FILE_TRACKING_JSON = "upload_tracking.json"  # JSON file tracking uploaded files (stored in HF repo)
FILE_METADATA_CSV = "metadata.csv"         # CSV file containing metadata for all audio files
FILE_EXTENSION_JSON = ".json"              # JSON file extension
FILE_EXTENSION_WAV = ".wav"                # WAV audio file extension

# --------------------------------------------------------------------------
# Tracking File Structure
# --------------------------------------------------------------------------
# Structure of the JSON tracking file stored in HuggingFace repo

TRACKING_KEY_VERSION = "version"                # Tracking file format version
TRACKING_KEY_UPLOADED_FILES = "uploaded_files"  # Dictionary of uploaded file records
TRACKING_KEY_TIMESTAMP = "upload_timestamp"     # When file was uploaded
TRACKING_KEY_API_RESPONSE = "api_response"      # Response from API
TRACKING_KEY_METADATA = "metadata_json"         # Metadata sent with upload
TRACKING_VERSION = "1.0"                        # Current tracking file version

# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------
# REST API endpoint paths for authentication, templates, and uploads

API_ENDPOINT_LOGIN = "/auth/login"                              # Login endpoint to get access token
API_ENDPOINT_TEMPLATES = "/metadata/templates"                  # List all available templates
API_ENDPOINT_TEMPLATE_DETAIL = "/metadata/templates/{template_id}"  # Get specific template details
API_ENDPOINT_UPLOAD = "/collection/upload/audio-with-chunks"    # Upload audio with chunk files

# --------------------------------------------------------------------------
# API Request/Response Fields
# --------------------------------------------------------------------------
# Field names used in API requests and responses

API_FIELD_ACCESS_TOKEN = "access_token"         # JWT token for authentication
API_FIELD_USERNAME = "username"                 # Username for login
API_FIELD_PASSWORD = "password"                 # Password for login
API_FIELD_TEMPLATE_ID = "template_id"           # ID of the metadata template
API_FIELD_LANGUAGE_TYPE = "language_type"       # Language of the audio (must be uppercase)
API_FIELD_UPLOAD_METADATA_STR = "upload_metadata_str"  # Metadata as JSON string
API_FIELD_FILES_METADATA = "files_metadata"     # File structure metadata as JSON string
API_FIELD_FILES = "files"                       # Multipart form field for file uploads

# --------------------------------------------------------------------------
# Template Field Names
# --------------------------------------------------------------------------
# Field names used in API template structure

TEMPLATE_FIELD_ID = "id"                    # Template ID field
TEMPLATE_FIELD_NAME = "name"                # Template name field
TEMPLATE_FIELD_FIELDS = "fields"            # Array of field definitions
TEMPLATE_FIELD_TYPE = "type"                # Field type (text, number, etc.)
TEMPLATE_FIELD_REQUIRED = "required"        # Whether field is required
TEMPLATE_FIELD_FIELD_NAME = "field_name"    # Alternative name for field name
TEMPLATE_FIELD_FIELD_ID = "field_id"        # Alternative name for field ID
TEMPLATE_FIELD_FIELD_TYPE = "field_type"    # Alternative name for field type

# --------------------------------------------------------------------------
# Metadata CSV Columns
# --------------------------------------------------------------------------
# Column names expected in the metadata.csv file from HuggingFace

CSV_COLUMN_MAIN_AUDIO_NAME = "main_audio_name"  # Unique audio file identifier (required)
CSV_COLUMN_LANGUAGE = "language"                 # Language code (required)
CSV_COLUMN_DOMAIN = "domain"                     # Domain/context of audio (required)
CSV_COLUMN_LOCATION = "location"                 # Dataset location (required)
CSV_COLUMN_SPEAKER_ID = "speaker_id"             # Speaker identifier (optional)
CSV_COLUMN_DURATION = "duration"                 # Audio duration (required)
CSV_COLUMN_AGE = "age"                           # Speaker age (optional)
CSV_COLUMN_GENDER = "gender"                     # Speaker gender (optional)

# --------------------------------------------------------------------------
# Metadata Field Names (for upload)
# --------------------------------------------------------------------------
# Field names used when preparing metadata for upload

FIELD_MAIN_AUDIO_NAME = "main_audio_name"
FIELD_LANGUAGE = "language"
FIELD_DOMAIN = "domain"
FIELD_LOCATION = "location"
FIELD_SPEAKER_ID = "speaker_id"
FIELD_DURATION = "duration"
FIELD_AGE = "age"
FIELD_GENDER = "gender"

# --------------------------------------------------------------------------
# Alternative Field Names for Template Mapping
# --------------------------------------------------------------------------
# Multiple possible names for each field to handle template variations
# The script tries all these variations when mapping CSV columns to template fields

FIELD_ALT_MAIN_AUDIO_NAME = ["main_audio_name", "Main Audio Name", "audio_name"]
FIELD_ALT_LANGUAGE = ["language", "Language"]
FIELD_ALT_DOMAIN = ["domain", "Domain", "Domain / Context", "Domain/Context", "context"]
FIELD_ALT_LOCATION = ["location", "Location", "Dataset Location", "dataset_location"]
FIELD_ALT_SPEAKER_ID = ["speaker_id", "Speaker ID", "speaker"]
FIELD_ALT_DURATION = ["duration", "Duration"]
FIELD_ALT_AGE = ["age", "Age"]
FIELD_ALT_GENDER = ["gender", "Gender"]

# --------------------------------------------------------------------------
# Language Mapping
# --------------------------------------------------------------------------
# Maps language codes (from CSV) to full uppercase names (required by API)
# The API requires language names in full uppercase format

LANGUAGE_MAPPING = {
    "nep": "NEPALI",        # Nepali language
    "nepali": "NEPALI",
    "mai": "MAITHILI",      # Maithili language
    "maithili": "MAITHILI",
    "hin": "HINDI",         # Hindi language
    "hindi": "HINDI",
    "eng": "ENGLISH",        # English language
    "english": "ENGLISH",
}

# --------------------------------------------------------------------------
# JSON Metadata Keys
# --------------------------------------------------------------------------
# Keys used in JSON metadata files that describe audio file structure

JSON_KEY_MAIN_FILE_NAME = "main_file_name"  # Name of the main audio file
JSON_KEY_CHUNK_FILES = "chunk_files"        # Array of chunk file information
JSON_KEY_NAME = "name"                      # Chunk file name
JSON_KEY_ORDER = "order"                    # Order/sequence of the chunk

# --------------------------------------------------------------------------
# HTTP Configuration
# --------------------------------------------------------------------------
# HTTP headers and content types

HTTP_HEADER_AUTHORIZATION = "Authorization"      # Authorization header name
HTTP_HEADER_BEARER_PREFIX = "Bearer "           # Bearer token prefix
HTTP_CONTENT_TYPE_AUDIO_WAV = "audio/wav"       # Content type for WAV files
HTTP_TIMEOUT_SECONDS = 600                      # Request timeout (10 minutes)

# --------------------------------------------------------------------------
# Log Message Prefixes
# --------------------------------------------------------------------------
# Prefixes for different types of log messages (for easy visual identification)

MSG_INFO_PREFIX = "[INFO]"      # Informational messages
MSG_OK_PREFIX = "[OK]"          # Success messages
MSG_WARN_PREFIX = "[WARN]"      # Warning messages
MSG_ERROR_PREFIX = "[ERROR]"    # Error messages
MSG_DEBUG_PREFIX = "[DEBUG]"    # Debug messages
MSG_SKIP_PREFIX = "[SKIP]"      # Skipped operations
MSG_SUCCESS_PREFIX = "[SUCCESS]" # Successful completion

# --------------------------------------------------------------------------
# Message Templates
# --------------------------------------------------------------------------
# Template strings for all messages (use .format() to fill in values)
# This centralizes all user-facing text for easy modification and i18n

MSG_TRACKING_FILE_DOWNLOADED = "Tracking file downloaded from HuggingFace"
MSG_TRACKING_FILE_NOT_FOUND = "No existing tracking file found - will create new one"
MSG_TRACKING_FILE_INITIALIZED = "Tracking data initialized"
MSG_TRACKING_FILE_UPLOADED = "Tracking file uploaded to HuggingFace"
MSG_MARKED_UPLOADED = "Marked {name} as uploaded in tracking file"
MSG_LOGGING_IN = "Logging in as {username}..."
MSG_LOGIN_SUCCESSFUL = "Login successful"
MSG_FETCHING_TEMPLATES = "Fetching available templates..."
MSG_FOUND_TEMPLATES = "Found {count} templates:"
MSG_TEMPLATE_ITEM = "  {num}. {name} (ID: {id})"
MSG_FETCHING_TEMPLATE_DETAILS = "Fetching template details for {id}..."
MSG_LOADING_TEMPLATE_BY_ID = "Loading template by ID: {id}"
MSG_SEARCHING_TEMPLATE_BY_NAME = "Searching for template by name: {name}"
MSG_TEMPLATE_NOT_FOUND = "Template '{name}' not found"
MSG_TEMPLATE_LOADED = "\nTemplate: {name}"
MSG_TEMPLATE_ID_LOADED = "Template ID: {id}"
MSG_FOUND_FIELDS = "Found {count} fields:"
MSG_FIELD_ITEM = "  - {name}: {id} ({type}) {required}"
MSG_FIELD_REQUIRED = "[REQUIRED]"
MSG_FIELD_OPTIONAL = "[OPTIONAL]"
MSG_SKIPPING_NONE_FIELD = "Skipping None field: {name}"
MSG_FIELD_NOT_FOUND = "Field '{name}' not found in template (tried: {tried}), skipping"
MSG_MAPPED_FIELD = "Mapped field: {name} -> {id} = '{value}'"
MSG_UPLOADING_AUDIO = "Uploading audio with {count} chunks..."
MSG_TEMPLATE_ID_DEBUG = "Template ID: {id}"
MSG_LANGUAGE_DEBUG = "Language: {lang}"
MSG_METADATA_DEBUG = "Metadata: {metadata}"
MSG_UPLOAD_SUCCESSFUL = "Upload successful: {result}"
MSG_UPLOAD_FAILED = "Upload failed with status {status}"
MSG_UPLOAD_RESPONSE_ERROR = "Response: {response}"
MSG_DOWNLOADING_METADATA = "Downloading metadata.csv from {repo}..."
MSG_ROW_SKIP_MISSING_FIELD = "Row {idx}: Skipping - missing {field}"
MSG_SKIPPED_ROWS = "Skipped {count} rows due to missing required fields"
MSG_LOADED_METADATA_ENTRIES = "Loaded {count} valid entries from metadata.csv"
MSG_DOWNLOADING_FILES = "Downloading files for {name}..."
MSG_DOWNLOADING_CHUNKS = "Downloading {count} chunks..."
MSG_DOWNLOAD_PROGRESS = "Progress: {current}/{total} chunks downloaded"
MSG_DOWNLOADED_CHUNKS = "Downloaded {count} chunks for {name}"
MSG_PROCESSING_BATCH = "\nProcessing batch {current}/{total}"
MSG_WAITING_BATCH = "Waiting {seconds} seconds before next batch..."
MSG_PROCESSING_AUDIO = "\nProcessing {name}"
MSG_UNKNOWN_LANGUAGE = "Unknown language code '{code}'. Supported: {supported}"
MSG_SKIPPING_AUDIO = "Skipping {name}"
MSG_LANGUAGE_MAPPING = "Language: {code} -> {full}"
MSG_FIELD_VALUES_DEBUG = "Field values (non-empty only): {values}"
MSG_NO_METADATA_MATCHED = "No metadata fields matched template fields!"
MSG_AVAILABLE_TEMPLATE_FIELDS = "Available template fields: {fields}"
MSG_UPLOADED_SUCCESS = "Uploaded {name}"
MSG_CLEANING_UP = "Cleaning up downloaded files for {name}..."
MSG_REMOVED_FOLDER = "Removed folder: {folder}"
MSG_DELETED_MAIN_AUDIO = "Deleted main audio: {name}"
MSG_CLEANED_UP_FILES = "Cleaned up {count} individual files"
MSG_CLEANUP_FAILED = "Cleanup failed (non-critical): {error}"
MSG_PROCESSING_FAILED = "Failed to process {name}: {type}: {error}"
MSG_ALREADY_UPLOADED = "{name} already uploaded"
MSG_SKIP_EMPTY_MAIN_AUDIO = "Skipping row with empty main_audio_name"
MSG_INTERRUPTED = "\nInterrupted by user"
MSG_MISSING_ENV_VARS = "Missing required environment variables: {vars}"
MSG_SET_ENV_VARS = "Please set them in your .env file"
MSG_NO_TEMPLATE_IN_ENV = "No TEMPLATE_ID or TEMPLATE_NAME in .env"
MSG_NO_TEMPLATES_FOUND = "No templates found in the system"
MSG_AUTO_SELECT_TEMPLATE = "\nAuto-selecting first template: {name}"
MSG_TOTAL_FILES_HF = "Total files in HuggingFace: {count}"
MSG_ALREADY_UPLOADED_COUNT = "Already uploaded: {count}"
MSG_NEW_FILES_TO_UPLOAD = "New files to upload: {count}"
MSG_NO_NEW_FILES = "No new files to upload. All files are up to date."
MSG_UPLOAD_SUMMARY = "Success: {success}/{total}"
MSG_FAILED_SUMMARY = "Failed: {failed}/{total}"
MSG_TOTAL_UPLOADED_ALL_TIME = "Total uploaded (all time): {count}"
MSG_HF_TOKEN_REQUIRED = "HF_TOKEN is required to upload tracking file to HuggingFace repository"

# --------------------------------------------------------------------------
# Section Headers
# --------------------------------------------------------------------------
# Formatted headers for different sections of the output

HEADER_MAIN = "\n" + "=" * 70 + "\nHuggingFace Audio Uploader - Batch Processing\n" + "=" * 70
HEADER_CONFIG = "HuggingFace Repo: {repo}\nAPI Base URL: {url}\nTemplate: {template}\nBatch Size: {size}\nBatch Delay: {delay}s"
HEADER_TEMPLATE_LOADING = "=" * 70 + "\nTEMPLATE LOADING\n" + "=" * 70 + "\n"
HEADER_LOADING_METADATA = "\n" + "=" * 70 + "\nLOADING METADATA FROM HUGGINGFACE\n" + "=" * 70 + "\n"
HEADER_PROCESSING_FILES = "\n" + "=" * 70 + "\nPROCESSING FILES\n" + "=" * 70 + "\n"
HEADER_UPLOAD_COMPLETE = "\n" + "=" * 70 + "\nUPLOAD COMPLETE\n" + "=" * 70

# --------------------------------------------------------------------------
# Miscellaneous Constants
# --------------------------------------------------------------------------

HF_REPO_TYPE_DATASET = "dataset"        # HuggingFace repository type
ENCODING_UTF8 = "utf-8"                 # File encoding for CSV files
EMPTY_STRING = ""                       # Empty string constant
TIMESTAMP_SUFFIX = "Z"                  # UTC timestamp suffix
UNNAMED_FIELD = "unnamed"               # Placeholder for unnamed fields
UNKNOWN_TYPE = "unknown"                # Placeholder for unknown field types
CHUNK_PROGRESS_INTERVAL = 10            # Show progress every N chunks

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
# These are pure helper functions that perform simple, reusable operations.
# They have no side effects and can be used throughout the codebase.

def load_environment() -> None:
    """
    Load environment variables from .env file.
    
    This function reads the .env file in the current directory and loads
    all key-value pairs into the environment. Must be called before
    attempting to read any environment variables.
    """
    load_dotenv()


def get_env(key: str, default: str = "") -> str:
    """
    Get environment variable value with a default fallback.
    
    Args:
        key: Environment variable name to retrieve
        default: Default value if the variable is not set
        
    Returns:
        Value of the environment variable or default if not found
    """
    return os.getenv(key, default)


# def get_env_int(key: str, default: str) -> int:
#     """
#     Get environment variable as an integer.
    
#     Args:
#         key: Environment variable name to retrieve
#         default: Default value as string (will be converted to int)
        
#     Returns:
#         Integer value of the environment variable
#     """
#     return int(get_env(key, default))


def create_directory(path: Path) -> Path:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path object representing the directory to create
        
    Returns:
        The same Path object (for chaining)
    """
    path.mkdir(exist_ok=True)
    return path


def print_info(message: str) -> None:
    """Print informational message with [INFO] prefix."""
    print(f"{MSG_INFO_PREFIX} {message}")


def print_ok(message: str) -> None:
    """Print success message with [OK] prefix."""
    print(f"{MSG_OK_PREFIX} {message}")


def print_warn(message: str) -> None:
    """Print warning message with [WARN] prefix."""
    print(f"{MSG_WARN_PREFIX} {message}")


def print_error(message: str) -> None:
    """Print error message with [ERROR] prefix."""
    print(f"{MSG_ERROR_PREFIX} {message}")


def print_debug(message: str) -> None:
    """Print debug message with [DEBUG] prefix."""
    print(f"{MSG_DEBUG_PREFIX} {message}")


def print_skip(message: str) -> None:
    """Print skip message with [SKIP] prefix."""
    print(f"{MSG_SKIP_PREFIX} {message}")


def print_success(message: str) -> None:
    """Print success message with [SUCCESS] prefix."""
    print(f"{MSG_SUCCESS_PREFIX} {message}")


def get_current_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns:
        ISO 8601 formatted timestamp string with 'Z' suffix (e.g., "2024-01-15T10:30:00.123456Z")
    """
    return datetime.utcnow().isoformat() + TIMESTAMP_SUFFIX


def normalize_language_code(language_code: str, mapping: Dict[str, str]) -> Optional[str]:
    """
    Normalize language code to full uppercase name using provided mapping.
    
    Args:
        language_code: Short language code (e.g., "nep", "hindi")
        mapping: Dictionary mapping codes to full names
        
    Returns:
        Full uppercase language name (e.g., "NEPALI") or None if not found
    """
    return mapping.get(language_code.lower())


def strip_or_empty(value: Optional[str]) -> str:
    """
    Strip whitespace from string or return empty string if None.
    
    Args:
        value: String to strip or None
        
    Returns:
        Stripped string or empty string if value was None
    """
    return value.strip() if value else EMPTY_STRING


# ============================================================================
# TRACKING FILE FUNCTIONS (HuggingFace Repository)
# ============================================================================
# These functions handle all tracking operations using a JSON file stored
# in the HuggingFace repository. This allows multiple team members to share
# upload history across different devices.

def create_empty_tracking_data() -> Dict:
    """
    Create empty tracking data structure.
    
    Returns:
        Dictionary with tracking file structure
    """
    return {
        TRACKING_KEY_VERSION: TRACKING_VERSION,
        TRACKING_KEY_UPLOADED_FILES: {}
    }


def download_tracking_file(repo_id: str, token: str) -> Dict:
    """
    Download tracking file from HuggingFace repository.
    
    Attempts to download the existing tracking file. If it doesn't exist yet,
    creates a new empty tracking structure.
    
    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token (required for reading private repos)
        
    Returns:
        Dictionary containing tracking data
    """
    try:
        # Try to download existing tracking file
        tracking_path = hf_hub_download(
            repo_id=repo_id,
            filename=FILE_TRACKING_JSON,
            token=token,
            repo_type=HF_REPO_TYPE_DATASET,
            local_dir=Path(".")
        )
        
        with open(tracking_path, 'r', encoding=ENCODING_UTF8) as f:
            tracking_data = json.load(f)
        
        print_ok(MSG_TRACKING_FILE_DOWNLOADED)
        return tracking_data
    
    except Exception as e:
        # File doesn't exist yet - create new tracking data
        print_warn(MSG_TRACKING_FILE_NOT_FOUND)
        tracking_data = create_empty_tracking_data()
        print_ok(MSG_TRACKING_FILE_INITIALIZED)
        return tracking_data


def upload_tracking_file(repo_id: str, token: str, tracking_data: Dict) -> None:
    """
    Upload tracking file to HuggingFace repository.
    
    Saves the tracking data to a local JSON file, then uploads it to the
    HuggingFace repository, overwriting the existing file.
    
    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token (required for writing)
        tracking_data: Dictionary containing tracking data to upload
    """
    # Save to local file first
    local_tracking_path = Path(FILE_TRACKING_JSON)
    with open(local_tracking_path, 'w', encoding=ENCODING_UTF8) as f:
        json.dump(tracking_data, f, indent=2, ensure_ascii=False)
    
    # Upload to HuggingFace repository
    api = HfApi(token=token)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            api.upload_file(
                path_or_fileobj=str(local_tracking_path),
                path_in_repo=FILE_TRACKING_JSON,
                repo_id=repo_id,
                repo_type=HF_REPO_TYPE_DATASET,
                commit_message=f"Update tracking file - {get_current_timestamp()}",
            )
            print_ok(MSG_TRACKING_FILE_UPLOADED)
            return # Success!
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Warning: Tracking upload failed (Attempt {attempt+1}). Retrying...")
                time.sleep(5)
            else:
                print_error(f"Failed to upload tracking file after {max_retries} attempts: {e}")


def check_if_uploaded(tracking_data: Dict, main_audio_name: str) -> bool:
    """
    Check if an audio file has already been uploaded.
    
    This prevents duplicate uploads by checking if a record exists
    in the tracking data for the given audio file name.
    
    Args:
        tracking_data: Dictionary containing tracking data
        main_audio_name: Unique name of the audio file to check
        
    Returns:
        True if the file has been uploaded before, False otherwise
    """
    uploaded_files = tracking_data.get(TRACKING_KEY_UPLOADED_FILES, {})
    return main_audio_name in uploaded_files


def mark_as_uploaded(
    tracking_data: Dict,
    main_audio_name: str,
    api_response: str,
    metadata: Dict
) -> None:
    """
    Mark an audio file as successfully uploaded in the tracking data.
    
    Adds or updates a record in the uploaded_files dictionary.
    Note: This only updates the in-memory tracking data. You must call
    upload_tracking_file() to persist changes to HuggingFace.
    
    Args:
        tracking_data: Dictionary containing tracking data
        main_audio_name: Unique name of the audio file
        api_response: JSON string of the API's response
        metadata: Dictionary of metadata that was sent with the upload
    """
    if TRACKING_KEY_UPLOADED_FILES not in tracking_data:
        tracking_data[TRACKING_KEY_UPLOADED_FILES] = {}
    
    tracking_data[TRACKING_KEY_UPLOADED_FILES][main_audio_name] = {
        TRACKING_KEY_TIMESTAMP: get_current_timestamp(),
        TRACKING_KEY_API_RESPONSE: api_response,
        TRACKING_KEY_METADATA: metadata
    }
    
    print_ok(MSG_MARKED_UPLOADED.format(name=main_audio_name))


def get_all_uploaded_files(tracking_data: Dict) -> Set[str]:
    """
    Get set of all uploaded file names.
    
    Retrieves all main_audio_name values from the tracking data. This is used
    to filter out already-uploaded files from the processing queue.
    
    Args:
        tracking_data: Dictionary containing tracking data
        
    Returns:
        Set of audio file names that have been uploaded
    """
    uploaded_files = tracking_data.get(TRACKING_KEY_UPLOADED_FILES, {})
    return set(uploaded_files.keys())


# ============================================================================
# API AUTHENTICATION FUNCTIONS
# ============================================================================
# These functions handle user authentication with the API server.
# They obtain and manage JWT access tokens for authenticated requests.

def perform_login(client: httpx.Client, base_url: str, username: str, password: str) -> str:
    """
    Authenticate with the API and get an access token.
    
    Sends username and password to the login endpoint and receives a
    JWT access token that will be used for all subsequent API requests.
    
    Args:
        client: HTTP client to use for the request
        base_url: Base URL of the API (e.g., "http://localhost:8000")
        username: Username for authentication
        password: Password for authentication
        
    Returns:
        JWT access token string
        
    Raises:
        httpx.HTTPStatusError: If login fails (invalid credentials, server error, etc.)
    """
    print_info(MSG_LOGGING_IN.format(username=username))
    response = client.post(
        f"{base_url}{API_ENDPOINT_LOGIN}",
        json={API_FIELD_USERNAME: username, API_FIELD_PASSWORD: password},
    )
    response.raise_for_status()  # Raise exception if status code is 4xx or 5xx
    data = response.json()
    access_token = data.get(API_FIELD_ACCESS_TOKEN)
    print_ok(MSG_LOGIN_SUCCESSFUL)
    return access_token


def create_auth_headers(access_token: str) -> Dict[str, str]:
    """
    Create HTTP headers with authorization token.
    
    Constructs the Authorization header required for authenticated API requests.
    The token is sent as a Bearer token in the format: "Bearer <token>"
    
    Args:
        access_token: JWT access token from login
        
    Returns:
        Dictionary containing the Authorization header
    """
    return {HTTP_HEADER_AUTHORIZATION: f"{HTTP_HEADER_BEARER_PREFIX}{access_token}"}


# ============================================================================
# API TEMPLATE FUNCTIONS
# ============================================================================
# These functions interact with the API's metadata template system.
# Templates define what metadata fields are required/optional for uploads.
# The script must map CSV columns to template fields before uploading.

def fetch_templates(client: httpx.Client, base_url: str, headers: Dict[str, str]) -> List[Dict]:
    """
    Fetch all available metadata templates from the API.
    
    Templates define the structure of metadata that can be attached to uploads.
    Each template has a unique ID, name, and set of fields.
    
    Args:
        client: HTTP client for making requests
        base_url: Base URL of the API
        headers: Authorization headers (with access token)
        
    Returns:
        List of template dictionaries, each containing id, name, and fields
    """
    # print_info(MSG_FETCHING_TEMPLATES)
    # response = client.get(f"{base_url}{API_ENDPOINT_TEMPLATES}", headers=headers)
    # response.raise_for_status()
    # return response.json()
    # # response = client.get(f"{base_url}{API_ENDPOINT_TEMPLATES}", headers=headers)
    # # response.raise_for_status()
    # # body = response.json()
    # # print(f"Template response type: {type(body)}")
    # # print(body)

    # # return body

    print_info(MSG_FETCHING_TEMPLATES)
    response = client.get(
        f"{base_url}{API_ENDPOINT_TEMPLATES}",
        params={"page_size": 100},
        headers=headers
    )
    response.raise_for_status()
    body = response.json()

    # Normalize: API may return list directly or wrapped in a key
    items = []
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        for key in ("data", "templates", "results", "items"):
            if key in body and isinstance(body[key], list):
                items = body[key]
                break
        else:
            if "id" in body and "name" in body:
                items = [body]

    # Filter to dicts only (API may return mixed types)
    return [t for t in items if isinstance(t, dict)]

def print_templates_list(templates: List[Dict]) -> None:
    """
    Print a numbered list of available templates.
    
    Displays templates in a user-friendly format showing their names and IDs.
    Useful for users to see what templates are available.
    
    Args:
        templates: List of template dictionaries from fetch_templates()
    """
    print_info(MSG_FOUND_TEMPLATES.format(count=len(templates)))
    for i, template in enumerate((t for t in templates if isinstance(t, dict)), 1):
        name = template.get(TEMPLATE_FIELD_NAME, UNNAMED_FIELD)
        template_id = template.get(TEMPLATE_FIELD_ID)
        print(MSG_TEMPLATE_ITEM.format(num=i, name=name, id=template_id))


def find_template_by_name(templates: List[Dict], template_name: str) -> Optional[Dict]:
    """
    Find a template by its name (case-insensitive search).
    
    Searches through the list of templates to find one matching the given name.
    The search is case-insensitive to be more user-friendly.
    
    Args:
        templates: List of template dictionaries
        template_name: Name of the template to find
        
    Returns:
        Template dictionary if found, None otherwise
    """
    for template in templates:
        if not isinstance(template, dict):
            continue
        if template.get(TEMPLATE_FIELD_NAME, EMPTY_STRING).lower() == template_name.lower():
            return template
    return None


def fetch_template_details(
    client: httpx.Client,
    base_url: str,
    template_id: str,
    headers: Dict[str, str]
) -> Dict:
    """
    Fetch detailed information about a specific template.
    
    Gets the complete template definition including all field specifications.
    This is needed to know what metadata fields are required for uploads.
    
    Args:
        client: HTTP client for making requests
        base_url: Base URL of the API
        template_id: Unique ID of the template to fetch
        headers: Authorization headers
        
    Returns:
        Complete template dictionary with fields array
    """
    print_info(MSG_FETCHING_TEMPLATE_DETAILS.format(id=template_id))
    endpoint = API_ENDPOINT_TEMPLATE_DETAIL.format(template_id=template_id)
    response = client.get(f"{base_url}{endpoint}", headers=headers)
    response.raise_for_status()
    return response.json()


def extract_field_mappings(template: Dict) -> Dict[str, str]:
    """
    Extract field name to field ID mappings from template.
    
    Creates a lookup dictionary to quickly find field IDs by field name.
    This is used when building metadata - we need to map friendly field names
    (like "speaker_id") to the template's field IDs (like "field_12345").
    
    The function handles multiple possible field name formats that different
    API versions might use (name vs field_name, id vs field_id).
    
    Args:
        template: Template dictionary with 'fields' array
        
    Returns:
        Dictionary mapping field names to field IDs
        Example: {"speaker_id": "field_12345", "language": "field_67890"}
    """
    field_mappings = {}
    fields = template.get(TEMPLATE_FIELD_FIELDS, [])
    
    for field in fields:
        # Try to get field name (handle both naming conventions)
        field_name = field.get(TEMPLATE_FIELD_NAME, field.get(TEMPLATE_FIELD_FIELD_NAME, UNNAMED_FIELD))
        # Try to get field ID (handle both naming conventions)
        field_id = field.get(TEMPLATE_FIELD_ID, field.get(TEMPLATE_FIELD_FIELD_ID))
        
        # Store mappings for both the name and ID
        if field_name and field_name != UNNAMED_FIELD:
            field_mappings[field_name] = field_id
        
        if field_id:
            field_mappings[field_id] = field_id
    
    return field_mappings


def print_template_info(template: Dict, field_mappings: Dict[str, str]) -> None:
    """
    Print detailed information about a loaded template.
    
    Displays the template name, ID, and all available fields with their
    types and whether they're required. Helps users understand what
    metadata they need to provide.
    
    Args:
        template: Template dictionary
        field_mappings: Field name to ID mappings from extract_field_mappings()
    """
    print_info(MSG_TEMPLATE_LOADED.format(name=template.get(TEMPLATE_FIELD_NAME, UNNAMED_FIELD)))
    print_info(MSG_TEMPLATE_ID_LOADED.format(id=template[TEMPLATE_FIELD_ID]))
    
    fields = template.get(TEMPLATE_FIELD_FIELDS, [])
    print_info(MSG_FOUND_FIELDS.format(count=len(fields)))
    
    for field in fields:
        field_name = field.get(TEMPLATE_FIELD_NAME, field.get(TEMPLATE_FIELD_FIELD_NAME, UNNAMED_FIELD))
        field_id = field.get(TEMPLATE_FIELD_ID, field.get(TEMPLATE_FIELD_FIELD_ID))
        field_type = field.get(TEMPLATE_FIELD_TYPE, field.get(TEMPLATE_FIELD_FIELD_TYPE, UNKNOWN_TYPE))
        required = field.get(TEMPLATE_FIELD_REQUIRED, False)
        
        req_marker = MSG_FIELD_REQUIRED if required else MSG_FIELD_OPTIONAL
        print(MSG_FIELD_ITEM.format(name=field_name, id=field_id, type=field_type, required=req_marker))
    
    print()


# ============================================================================
# METADATA BUILDING FUNCTIONS
# ============================================================================
# These functions handle the conversion of CSV metadata into the format
# required by the API. They map friendly field names to template field IDs
# and build the metadata structure expected by the upload endpoint.

def get_field_name_alternatives() -> Dict[str, List[str]]:
    """
    Get mapping of standard field names to their possible template variations.
    
    Different templates might use different names for the same field.
    For example, "main_audio_name" might be called "Main Audio Name" or "audio_name".
    This function returns all possible variations we should try when mapping fields.
    
    Returns:
        Dictionary mapping standard field names to lists of alternative names
        Example: {"speaker_id": ["speaker_id", "Speaker ID", "speaker"]}
    """
    return {
        FIELD_MAIN_AUDIO_NAME: FIELD_ALT_MAIN_AUDIO_NAME,
        FIELD_LANGUAGE: FIELD_ALT_LANGUAGE,
        FIELD_DOMAIN: FIELD_ALT_DOMAIN,
        FIELD_LOCATION: FIELD_ALT_LOCATION,
        FIELD_SPEAKER_ID: FIELD_ALT_SPEAKER_ID,
        FIELD_DURATION: FIELD_ALT_DURATION,
        FIELD_AGE: FIELD_ALT_AGE,
        FIELD_GENDER: FIELD_ALT_GENDER,
    }


def find_field_id(
    field_name: str,
    template_fields: Dict[str, str],
    field_mappings: Dict[str, List[str]]
) -> Optional[str]:
    """
    Find the template field ID for a given field name.
    
    Tries multiple possible names for each field to handle template variations.
    First tries exact match, then case-insensitive match.
    
    Example:
        - Looking for "speaker_id"
        - Tries: "speaker_id", "Speaker ID", "speaker"
        - Returns the field ID if any variation matches
    
    Args:
        field_name: Standard field name from our CSV (e.g., "speaker_id")
        template_fields: Dictionary of template field names to IDs
        field_mappings: Dictionary of field names to their alternative names
        
    Returns:
        Template field ID if found, None if no match
    """
    # Get all possible names for this field
    possible_names = field_mappings.get(field_name, [field_name])
    
    for possible_name in possible_names:
        # Try exact match first
        if possible_name in template_fields:
            return template_fields[possible_name]
        
        # Try case-insensitive match
        for template_field_name, template_field_id in template_fields.items():
            if template_field_name.lower() == possible_name.lower():
                return template_field_id
    
    return None


def build_upload_metadata(
    field_values: Dict[str, str],
    template_fields: Dict[str, str]
) -> List[Dict]:
    """
    Build upload metadata from field values.
    
    Converts our friendly field names and values into the format expected
    by the API. The API requires:
    - field_id: The template's field ID (not the friendly name)
    - field_value: The value as a string
    
    This function:
    1. Takes field values from CSV (e.g., {"speaker_id": "SPK001"})
    2. Maps field names to template field IDs
    3. Builds array of {field_id, field_value} objects
    4. Includes all fields even if empty (API requirement)
    
    Args:
        field_values: Dictionary of field names to values from CSV
        template_fields: Dictionary of template field names to IDs
        
    Returns:
        List of metadata objects for upload
        Example: [{"field_id": "field_123", "field_value": "SPK001"}]
    """
    field_mappings = get_field_name_alternatives()
    metadata = []
    
    for field_name, field_value in field_values.items():
        # Skip None values (but include empty strings - API requirement)
        if field_value is None:
            print_debug(MSG_SKIPPING_NONE_FIELD.format(name=field_name))
            continue
        
        # Find the template field ID for this field
        field_id = find_field_id(field_name, template_fields, field_mappings)
        
        if not field_id:
            # Field not found in template - log warning and skip
            possible_names = field_mappings.get(field_name, [field_name])
            print_warn(MSG_FIELD_NOT_FOUND.format(name=field_name, tried=possible_names))
            continue
        
        # Add to metadata
        metadata.append({
            TEMPLATE_FIELD_FIELD_ID: field_id,
            "field_value": str(field_value)  # Ensure value is string
        })
        print_debug(MSG_MAPPED_FIELD.format(name=field_name, id=field_id, value=field_value))
    
    return metadata


# ============================================================================
# API UPLOAD FUNCTIONS
# ============================================================================
# These functions handle the complex process of uploading audio files to the API.
# Audio uploads consist of:
# 1. Main audio file (full recording)
# 2. Multiple chunk files (16-second segments)
# 3. Metadata (speaker info, language, domain, etc.)
# All files are sent as multipart/form-data in a single request.

def prepare_files_metadata(
    main_file_name: str,
    chunk_files: List[Tuple[Path, int]]
) -> Dict:
    """
    Prepare metadata describing the structure of files being uploaded.
    
    The API needs to know:
    - The name of the main audio file
    - The names and order of all chunk files
    
    This metadata helps the API organize and validate the uploaded files.
    
    Args:
        main_file_name: Name of the main audio file
        chunk_files: List of (chunk_path, order) tuples
        
    Returns:
        Dictionary with main_file_name and chunk_files array
    """
    chunk_files_metadata = [
        {JSON_KEY_NAME: chunk_path.name, JSON_KEY_ORDER: order}
        for chunk_path, order in sorted(chunk_files, key=lambda x: x[1])
    ]
    
    return {
        JSON_KEY_MAIN_FILE_NAME: main_file_name,
        JSON_KEY_CHUNK_FILES: chunk_files_metadata,
    }


def prepare_upload_data(
    template_id: str,
    language_type: str,
    upload_metadata: List[Dict],
    files_metadata: Dict
) -> Dict:
    """
    Prepare the data payload for the upload request.
    
    Creates the form data that will be sent with the file upload.
    All metadata is sent as JSON strings in form fields.
    
    Args:
        template_id: ID of the metadata template being used
        language_type: Full uppercase language name (e.g., "NEPALI")
        upload_metadata: Array of field_id/field_value pairs
        files_metadata: File structure metadata from prepare_files_metadata()
        
    Returns:
        Dictionary of form field names to values
    """
    return {
        API_FIELD_TEMPLATE_ID: template_id,
        API_FIELD_LANGUAGE_TYPE: language_type,
        API_FIELD_UPLOAD_METADATA_STR: json.dumps(upload_metadata),
        API_FIELD_FILES_METADATA: json.dumps(files_metadata),
    }


def prepare_upload_files(
    main_file_path: Path,
    chunk_files: List[Tuple[Path, int]]
) -> List[Tuple[str, Tuple]]:
    """
    Prepare file handles for multipart upload.
    
    Opens all audio files (main + chunks) and prepares them for upload.
    Each file is a tuple of (field_name, (filename, file_object, content_type)).
    
    IMPORTANT: Files must be closed after upload using close_upload_files()
    
    Args:
        main_file_path: Path to the main audio file
        chunk_files: List of (chunk_path, order) tuples
        
    Returns:
        List of file tuples for httpx multipart upload
    """
    files_to_upload = []
    
    # Add main audio file
    files_to_upload.append(
        (API_FIELD_FILES, (main_file_path.name, open(main_file_path, "rb"), HTTP_CONTENT_TYPE_AUDIO_WAV))
    )
    
    # Add all chunk files in order
    for chunk_path, order in sorted(chunk_files, key=lambda x: x[1]):
        files_to_upload.append(
            (API_FIELD_FILES, (chunk_path.name, open(chunk_path, "rb"), HTTP_CONTENT_TYPE_AUDIO_WAV))
        )
    
    return files_to_upload


def close_upload_files(files_to_upload: List[Tuple[str, Tuple]]) -> None:
    """
    Close all opened file handles.
    
    MUST be called after upload to prevent file handle leaks.
    Uses a finally block in perform_upload() to ensure cleanup.
    
    Args:
        files_to_upload: List of file tuples from prepare_upload_files()
    """
    for _, file_tuple in files_to_upload:
        file_tuple[1].close()  # Close the file object


def perform_upload(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    data: Dict,
    files_to_upload: List[Tuple[str, Tuple]]
) -> Dict:
    """
    Perform the actual HTTP upload request.
    
    Sends all files and metadata to the API in a single multipart/form-data request.
    Handles errors and ensures file cleanup.
    
    Args:
        client: HTTP client for making the request
        base_url: Base URL of the API
        headers: Authorization headers
        data: Form data (template_id, language, metadata)
        files_to_upload: Opened file handles from prepare_upload_files()
        
    Returns:
        API response as dictionary
        
    Raises:
        httpx.HTTPStatusError: If upload fails (4xx or 5xx status)
    """
    try:
        response = client.post(
            f"{base_url}{API_ENDPOINT_UPLOAD}",
            data=data,
            files=files_to_upload,
            headers=headers,
        )
        response.raise_for_status()  # Raise exception for error status codes
        result = response.json()
        print_ok(MSG_UPLOAD_SUCCESSFUL.format(result=result))
        return result
    except httpx.HTTPStatusError as e:
        # Log detailed error information
        print_error(MSG_UPLOAD_FAILED.format(status=e.response.status_code))
        print_error(MSG_UPLOAD_RESPONSE_ERROR.format(response=e.response.text))
        raise
    finally:
        # Always close files, even if upload fails
        close_upload_files(files_to_upload)


# ============================================================================
# HUGGINGFACE FUNCTIONS
# ============================================================================
# These functions download files from a HuggingFace dataset repository.
# The expected repository structure is:
# - metadata.csv (file list with speaker info, language, domain, etc.)
# - upload_tracking.json (tracking file - created by this script)
# - audio/{audio_name}/{audio_name}.json (file structure metadata)
# - audio/{audio_name}/{audio_name}.wav (main audio file)
# - audio/{audio_name}/{audio_name}_chunk_001.wav, _chunk_002.wav, ... (16s chunks)

def download_file_from_hf(
    repo_id: str,
    filename: str,
    token: Optional[str],
    local_dir: Path
) -> Path:
    """
    Download a single file from HuggingFace repository.
    
    Uses the HuggingFace Hub library to download files. Files are cached
    locally to avoid re-downloading.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "user/dataset-name")
        filename: Path to file within the repository
        token: HuggingFace API token (optional for public repos)
        local_dir: Local directory to download to
        
    Returns:
        Path object pointing to the downloaded file
    """
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        repo_type=HF_REPO_TYPE_DATASET,  # Specify this is a dataset, not a model
        local_dir=local_dir
    )
    return Path(file_path)


def parse_metadata_csv(csv_path: Path) -> List[Dict]:
    """
    Parse metadata CSV file and return list of valid entries.
    
    The CSV should have these columns:
    - main_audio_name (required): Unique identifier for the audio file
    - language (required): Language code (e.g., "nep", "hindi")
    - domain (required): Domain or context (e.g., "news", "conversation")
    - location (required): Dataset location
    - speaker_id (optional): Speaker identifier
    - duration (required): Audio duration
    - age (optional): Speaker age
    - gender (optional): Speaker gender
    
    Rows missing required fields are skipped and logged.
    
    Args:
        csv_path: Path to the metadata.csv file
        
    Returns:
        List of dictionaries, one per valid CSV row
    """
    metadata_list = []
    skipped_count = 0
    
    with open(csv_path, 'r', encoding=ENCODING_UTF8) as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            # Extract and validate required fields
            main_audio_name = strip_or_empty(row.get(CSV_COLUMN_MAIN_AUDIO_NAME))
            language = strip_or_empty(row.get(CSV_COLUMN_LANGUAGE))
            domain = strip_or_empty(row.get(CSV_COLUMN_DOMAIN))
            location = strip_or_empty(row.get(CSV_COLUMN_LOCATION))
            duration = strip_or_empty(row.get(CSV_COLUMN_DURATION))
            
            # Validate each required field
            if not main_audio_name:
                print_warn(MSG_ROW_SKIP_MISSING_FIELD.format(idx=idx, field=CSV_COLUMN_MAIN_AUDIO_NAME))
                skipped_count += 1
                continue
            
            if not language:
                print_warn(MSG_ROW_SKIP_MISSING_FIELD.format(idx=f"{idx} ({main_audio_name})", field=CSV_COLUMN_LANGUAGE))
                skipped_count += 1
                continue
            

            # Build metadata entry (include optional fields as empty strings if not provided)
            metadata_list.append({
                CSV_COLUMN_MAIN_AUDIO_NAME: main_audio_name,
                CSV_COLUMN_LANGUAGE: language,
                CSV_COLUMN_DOMAIN: domain,
                CSV_COLUMN_LOCATION: location,
                CSV_COLUMN_SPEAKER_ID: strip_or_empty(row.get(CSV_COLUMN_SPEAKER_ID)),
                CSV_COLUMN_DURATION: duration,
                CSV_COLUMN_AGE: strip_or_empty(row.get(CSV_COLUMN_AGE)),
                CSV_COLUMN_GENDER: strip_or_empty(row.get(CSV_COLUMN_GENDER)),
            })
    
    if skipped_count > 0:
        print_warn(MSG_SKIPPED_ROWS.format(count=skipped_count))
    print_ok(MSG_LOADED_METADATA_ENTRIES.format(count=len(metadata_list)))
    
    return metadata_list


def load_json_metadata(json_path: Path) -> Dict:
    """
    Load JSON metadata file describing audio file structure.
    
    The JSON file contains:
    - main_file_name: Name of the main audio file
    - chunk_files: Array of {name, order} objects for chunks
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary with file structure information
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def download_chunks_with_progress(
    repo_id: str,
    audio_folder: str,
    chunk_files_info: List[Dict],
    token: Optional[str],
    local_dir: Path
) -> List[Tuple[Path, int]]:
    """
    Download chunk files with progress reporting.
    
    Downloads all chunk files for an audio file, showing progress
    every 10 chunks to keep the user informed without spam.
    
    Args:
        repo_id: HuggingFace repository ID
        audio_folder: Folder path within repo (e.g., "audio/audio_001")
        chunk_files_info: List of chunk metadata from JSON
        token: HuggingFace API token
        local_dir: Local download directory
        
    Returns:
        List of (chunk_path, order) tuples
    """
    chunk_files = []
    total_chunks = len(chunk_files_info)
    
    print_info(MSG_DOWNLOADING_CHUNKS.format(count=total_chunks))
    
    for idx, chunk_info in enumerate(chunk_files_info, 1):
        chunk_filename = f"{audio_folder}/{chunk_info[JSON_KEY_NAME]}"
        
        # Show progress every 10 chunks (or at the end)
        if idx % CHUNK_PROGRESS_INTERVAL == 0 or idx == total_chunks:
            print_info(MSG_DOWNLOAD_PROGRESS.format(current=idx, total=total_chunks))
        
        chunk_path = download_file_from_hf(repo_id, chunk_filename, token, local_dir)
        chunk_files.append((chunk_path, chunk_info[JSON_KEY_ORDER]))
    
    return chunk_files


# ============================================================================
# FILE CLEANUP FUNCTIONS
# ============================================================================
# These functions handle cleanup of downloaded files after successful upload.
# This prevents disk space from filling up with temporary files.

def remove_directory(directory: Path) -> bool:
    """
    Remove directory and all its contents.
    
    Recursively deletes a directory tree. Used to clean up
    the entire audio folder after upload.
    
    Args:
        directory: Path to directory to remove
        
    Returns:
        True if directory was removed, False if it didn't exist
    """
    if directory.exists() and directory.is_dir():
        shutil.rmtree(directory)
        print_ok(MSG_REMOVED_FOLDER.format(folder=directory))
        return True
    return False


def remove_individual_files(
    main_audio_path: Path,
    chunk_files: List[Tuple[Path, int]]
) -> int:
    """
    Remove individual files (fallback if directory removal fails).
    
    Deletes the main audio file and all chunk files one by one.
    This is a fallback if remove_directory() fails.
    
    Args:
        main_audio_path: Path to main audio file
        chunk_files: List of (chunk_path, order) tuples
        
    Returns:
        Number of files successfully deleted
    """
    deleted_count = 0
    
    # Remove main audio file
    if main_audio_path.exists():
        main_audio_path.unlink()
        print_debug(MSG_DELETED_MAIN_AUDIO.format(name=main_audio_path.name))
        deleted_count += 1
    
    # Remove all chunk files
    for chunk_path, _ in chunk_files:
        if chunk_path.exists():
            chunk_path.unlink()
            deleted_count += 1
    
    return deleted_count


def cleanup_downloaded_files(
    main_audio_name: str,
    main_audio_path: Path,
    chunk_files: List[Tuple[Path, int]],
    workdir: Path
) -> None:
    """
    Clean up downloaded files after successful upload.
    
    Removes all temporary files to free up disk space. Tries to remove
    the entire directory first, falls back to individual file deletion
    if that fails. Cleanup errors are non-fatal - they're logged but
    don't stop the upload process.
    
    Args:
        main_audio_name: Name of the audio file (for logging)
        main_audio_path: Path to main audio file
        chunk_files: List of (chunk_path, order) tuples
        workdir: Working directory containing the audio folder
    """
    try:
        print_info(MSG_CLEANING_UP.format(name=main_audio_name))
        
        # Try to remove entire folder (most efficient)
        audio_folder = workdir / DIR_AUDIO / main_audio_name
        
        if not remove_directory(audio_folder):
            # Fallback: remove individual files
            deleted_count = remove_individual_files(main_audio_path, chunk_files)
            print_ok(MSG_CLEANED_UP_FILES.format(count=deleted_count))
    
    except Exception as cleanup_error:
        # Cleanup errors are non-critical - log and continue
        print_warn(MSG_CLEANUP_FAILED.format(error=cleanup_error))


# ============================================================================
# FIELD VALUES PREPARATION
# ============================================================================
# This function extracts and normalizes field values from CSV metadata rows.

def prepare_field_values(metadata_row: Dict) -> Dict[str, str]:
    """
    Prepare field values from metadata CSV row.
    
    Extracts all metadata fields from a CSV row and normalizes them
    (strips whitespace, converts None to empty string). All fields are
    included even if empty - the API requires this.
    
    Args:
        metadata_row: Dictionary from CSV DictReader
        
    Returns:
        Dictionary of normalized field values
        Example: {
            "main_audio_name": "audio_001",
            "language": "nep",
            "speaker_id": "SPK001",
            "age": ""  # Empty but included
        }
    """
    return {
        FIELD_MAIN_AUDIO_NAME: strip_or_empty(metadata_row.get(CSV_COLUMN_MAIN_AUDIO_NAME)),
        FIELD_LANGUAGE: strip_or_empty(metadata_row.get(CSV_COLUMN_LANGUAGE)).capitalize(),
        FIELD_DOMAIN: strip_or_empty(metadata_row.get(CSV_COLUMN_DOMAIN)).capitalize(),
        FIELD_LOCATION: strip_or_empty(metadata_row.get(CSV_COLUMN_LOCATION)),
        FIELD_SPEAKER_ID: strip_or_empty(metadata_row.get(CSV_COLUMN_SPEAKER_ID)),
        FIELD_DURATION: strip_or_empty(metadata_row.get(CSV_COLUMN_DURATION)),
        FIELD_AGE: strip_or_empty(metadata_row.get(CSV_COLUMN_AGE)),
        FIELD_GENDER: strip_or_empty(metadata_row.get(CSV_COLUMN_GENDER)),
    }


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================
# These are the high-level orchestration functions that tie everything together.
# They coordinate downloads, metadata preparation, uploads, and cleanup.

def download_audio_files_from_hf(
    repo_id: str,
    main_audio_name: str,
    token: Optional[str],
    local_dir: Path
) -> Tuple[Path, List[Tuple[Path, int]], Dict]:
    """
    Download main audio, chunks, and JSON metadata for a specific audio file.
    
    This is a memory-efficient download process:
    1. Download small JSON file first to get structure
    2. Download main audio file
    3. Download chunks with progress reporting
    
    All files for this audio are downloaded to:
    {local_dir}/audio/{main_audio_name}/
    
    Args:
        repo_id: HuggingFace repository ID
        main_audio_name: Unique name of the audio file
        token: HuggingFace API token
        local_dir: Local directory to download to
        
    Returns:
        Tuple of (main_audio_path, chunk_files, audio_metadata)
        - main_audio_path: Path to downloaded main audio file
        - chunk_files: List of (chunk_path, order) tuples
        - audio_metadata: Dict from JSON with file structure info
    """
    audio_folder = f"{DIR_AUDIO}/{main_audio_name}"
    
    print_info(MSG_DOWNLOADING_FILES.format(name=main_audio_name))
    
    # Step 1: Download JSON metadata to know what files we need
    json_filename = f"{audio_folder}/{main_audio_name}{FILE_EXTENSION_JSON}"
    json_path = download_file_from_hf(repo_id, json_filename, token, local_dir)
    audio_metadata = load_json_metadata(json_path)
    
    # Step 2: Download main audio file
    main_audio_filename = f"{audio_folder}/{audio_metadata[JSON_KEY_MAIN_FILE_NAME]}"
    main_audio_path = download_file_from_hf(repo_id, main_audio_filename, token, local_dir)
    
    # Step 3: Download all chunk files
    chunk_files = download_chunks_with_progress(
        repo_id,
        audio_folder,
        audio_metadata[JSON_KEY_CHUNK_FILES],
        token,
        local_dir
    )
    
    print_ok(MSG_DOWNLOADED_CHUNKS.format(count=len(chunk_files), name=main_audio_name))
    
    return main_audio_path, chunk_files, audio_metadata


def upload_audio_to_api(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    template_id: str,
    main_file_path: Path,
    chunk_files: List[Tuple[Path, int]],
    language_type: str,
    upload_metadata: List[Dict]
) -> Dict:
    """
    Upload audio file with chunks to the API.
    
    This is a high-level wrapper that:
    1. Prepares file structure metadata
    2. Prepares upload form data
    3. Opens file handles
    4. Performs the upload
    5. Closes file handles (even if upload fails)
    
    Args:
        client: HTTP client for requests
        base_url: API base URL
        headers: Authorization headers
        template_id: Metadata template ID
        main_file_path: Path to main audio file
        chunk_files: List of (chunk_path, order) tuples
        language_type: Full uppercase language name (e.g., "NEPALI")
        upload_metadata: List of {field_id, field_value} dicts
        
    Returns:
        API response dictionary
    """
    print_info(MSG_UPLOADING_AUDIO.format(count=len(chunk_files)))
    
    # Log debug information
    print_debug(MSG_TEMPLATE_ID_DEBUG.format(id=template_id))
    print_debug(MSG_LANGUAGE_DEBUG.format(lang=language_type))
    print_debug(MSG_METADATA_DEBUG.format(metadata=json.dumps(upload_metadata)))
    
    # Prepare all components for upload
    files_metadata = prepare_files_metadata(main_file_path.name, chunk_files)
    data = prepare_upload_data(template_id, language_type, upload_metadata, files_metadata)
    files_to_upload = prepare_upload_files(main_file_path, chunk_files)
    
    # Perform the upload (files will be closed in finally block)
    return perform_upload(client, base_url, headers, data, files_to_upload)


def process_single_audio_file(
    metadata_row: Dict,
    repo_id: str,
    hf_token: str,
    workdir: Path,
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    template_id: str,
    template_fields: Dict[str, str],
    tracking_data: Dict,
    remote_file_list: Set[str]
) -> None:
    """
    Process a single audio file: download from HuggingFace, upload to API, track, cleanup.
    
    This is the main processing function for each audio file. It orchestrates the
    complete workflow:
    1. Check if already uploaded (skip if so)
    2. Download files from HuggingFace
    3. Normalize language code
    4. Prepare metadata fields
    5. Build upload metadata
    6. Upload to API
    7. Mark as uploaded in tracking data
    8. Upload updated tracking file to HuggingFace
    9. Clean up downloaded files
    
    Any errors are logged and re-raised (will be caught by batch processor).
    
    Args:
        metadata_row: Row from metadata.csv with speaker info, language, etc.
        repo_id: HuggingFace repository ID
        hf_token: HuggingFace API token (required for uploading tracking file)
        workdir: Working directory for downloads
        client: HTTP client for API requests
        base_url: API base URL
        headers: Authorization headers
        template_id: Metadata template ID
        template_fields: Field name to ID mappings
        tracking_data: Tracking data dictionary (will be modified)
        
    Raises:
        Exception: Any error during processing (download, upload, etc.)
    """
    main_audio_name = metadata_row.get(CSV_COLUMN_MAIN_AUDIO_NAME, EMPTY_STRING)
    
    # Validate we have a main_audio_name
    if not main_audio_name:
        print_warn(MSG_SKIP_EMPTY_MAIN_AUDIO)
        return

    remote_path = f"data/{main_audio_name}" 

    if remote_path in remote_file_list:
        print_skip(f"Verified: {main_audio_name} exists on HF server. Updating local tracker.")
        # Mark it in the local tracker so the JSON eventually syncs
        mark_as_uploaded(tracking_data, main_audio_name, "Verified on Server", {})
        return
    
    # Check if already uploaded (avoid duplicates)
    if check_if_uploaded(tracking_data, main_audio_name):
        print_skip(MSG_ALREADY_UPLOADED.format(name=main_audio_name))
        return
    
    try:
        print_info(MSG_PROCESSING_AUDIO.format(name=main_audio_name))
        
        # Step 1: Download files from HuggingFace
        main_audio_path, chunk_files, audio_metadata = download_audio_files_from_hf(
            repo_id, main_audio_name, hf_token, workdir
        )
        
        # Step 2: Normalize language code to full uppercase name
        language_code = metadata_row[CSV_COLUMN_LANGUAGE].lower()
        language_full = normalize_language_code(language_code, LANGUAGE_MAPPING)
        
        if not language_full:
            # Language code not in mapping - log error and skip
            print_error(MSG_UNKNOWN_LANGUAGE.format(
                code=language_code,
                supported=list(LANGUAGE_MAPPING.keys())
            ))
            print_error(MSG_SKIPPING_AUDIO.format(name=main_audio_name))
            return
        
        print_info(MSG_LANGUAGE_MAPPING.format(code=language_code, full=language_full))
        
        # Step 3: Prepare field values from CSV row
        field_values = prepare_field_values(metadata_row)
        print_debug(MSG_FIELD_VALUES_DEBUG.format(values=field_values))
        
        # Step 4: Build upload metadata (map fields to template)
        upload_metadata = build_upload_metadata(field_values, template_fields)
        
        if not upload_metadata:
            # No fields matched - log error and skip
            print_error(MSG_NO_METADATA_MATCHED)
            print_error(MSG_AVAILABLE_TEMPLATE_FIELDS.format(fields=list(template_fields.keys())))
            print_error(MSG_SKIPPING_AUDIO.format(name=main_audio_name))
            return
        
        # Step 5: Upload to API
        result = upload_audio_to_api(
            client, base_url, headers, template_id,
            main_audio_path, chunk_files, language_full, upload_metadata
        )
        
        # Step 6: Mark as successfully uploaded in tracking data
        mark_as_uploaded(tracking_data, main_audio_name, json.dumps(result), field_values)
        
        # Step 7: Upload updated tracking file to HuggingFace
        upload_tracking_file(repo_id, hf_token, tracking_data)
        
        print_success(MSG_UPLOADED_SUCCESS.format(name=main_audio_name))
        
        # Step 8: Clean up downloaded files
        cleanup_downloaded_files(main_audio_name, main_audio_path, chunk_files, workdir)
    
    except Exception as e:
        # Log error with full details
        print_error(MSG_PROCESSING_FAILED.format(
            name=main_audio_name,
            type=type(e).__name__,
            error=e
        ))
        traceback.print_exc()  # Print full stack trace for debugging
        raise  # Re-raise to let batch processor count it as a failure


# ============================================================================
# BATCH PROCESSING
# ============================================================================
# Processes multiple items in batches with rate limiting to avoid overwhelming
# the API server and to manage resources efficiently.

def process_batch_with_delay(
    items: List,
    process_func: Callable,
    batch_size: int,
    delay_seconds: int
) -> Tuple[int, int]:
    """
    Process items in batches with delay between batches.
    
    This prevents overwhelming the API server and helps manage resources.
    Each batch is processed sequentially, with a delay before starting the
    next batch (except after the last batch).
    
    Tracks success and failure counts. Failures don't stop processing -
    the function continues with remaining items.
    
    Args:
        items: List of items to process
        process_func: Function to call for each item
        batch_size: Number of items per batch
        delay_seconds: Seconds to wait between batches
        
    Returns:
        Tuple of (success_count, fail_count)
        
    Raises:
        KeyboardInterrupt: If user presses Ctrl+C (propagated to caller)
    """
    total = len(items)
    success_count = 0
    fail_count = 0
    
    # Process items in batches
    for i in range(0, total, batch_size):
        batch = items[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size  # Ceiling division
        
        print_info(MSG_PROCESSING_BATCH.format(current=batch_num, total=total_batches))
        
        # Process each item in the batch
        for item in batch:
            try:
                process_func(item)
                success_count += 1
            except KeyboardInterrupt:
                # User wants to stop - propagate to caller
                raise
            except Exception:
                # Log error but continue processing
                fail_count += 1
        
        # Wait before next batch (except after last batch)
        if i + batch_size < total:
            print_info(MSG_WAITING_BATCH.format(seconds=delay_seconds))
            time.sleep(delay_seconds)
    
    return success_count, fail_count


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================
# Functions to load and validate configuration from environment variables.

def load_configuration() -> Dict:
    """
    Load all configuration from environment variables.
    
    Reads .env file and extracts all configuration values with defaults.
    Returns a dictionary with all config values needed by the application.
    
    Returns:
        Dictionary with keys:
        - hf_repo_id: HuggingFace repository ID
        - hf_token: HuggingFace API token (REQUIRED for uploading tracking file)
        - api_base_url: API server URL
        - api_username: API username for authentication
        - api_password: API password for authentication
        - template_name: Metadata template name (may be None)
        - template_id: Metadata template ID (may be None)
        - batch_size: Number of files per batch
        - batch_delay_seconds: Delay between batches
    """
    load_environment()
    
    return {
        "hf_repo_id": get_env(ENV_HF_REPO_ID),
        "hf_token": get_env(ENV_HF_TOKEN),
        "api_base_url": get_env(ENV_API_BASE_URL),
        "api_username": get_env(ENV_API_USERNAME),
        "api_password": get_env(ENV_API_PASSWORD),
        "template_name": get_env(ENV_TEMPLATE_NAME),
        "template_id": get_env(ENV_TEMPLATE_ID),
        "batch_size": int(get_env(ENV_BATCH_SIZE)),
        "batch_delay_seconds": int(get_env(ENV_BATCH_DELAY_SECONDS)),
        "hf_skip_remote_file_list": get_env(ENV_HF_SKIP_REMOTE_FILE_LIST).lower() in ("1", "true", "yes"),
    }


def validate_configuration(config: Dict) -> List[str]:
    """
    Validate that all required configuration values are present.
    
    Checks for required environment variables and returns list of
    missing ones. The application cannot proceed without these.
    
    NOTE: HF_TOKEN is now REQUIRED for uploading the tracking file.
    
    Args:
        config: Configuration dictionary from load_configuration()
        
    Returns:
        List of missing required config keys (empty if all present)
    """
    required_keys = ["hf_repo_id", "hf_token", "api_username", "api_password"]
    return [key for key in required_keys if not config.get(key)]


# ============================================================================
# TEMPLATE LOADING
# ============================================================================
# Functions to load metadata templates from the API. Templates can be loaded
# by ID or by name, with automatic selection as a fallback.

def load_template_for_client(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str],
    template_id: Optional[str],
    template_name: Optional[str]
) -> Tuple[Dict, Dict[str, str]]:
    """
    Load template by ID or name and return template data with field mappings.
    
    This function handles two ways of specifying a template:
    1. By ID (direct lookup)
    2. By name (search, then lookup)
    
    Args:
        client: HTTP client for API requests
        base_url: API base URL
        headers: Authorization headers
        template_id: Template ID if loading by ID (takes precedence)
        template_name: Template name if loading by name
        
    Returns:
        Tuple of (template_dict, field_mappings_dict)
        - template_dict: Complete template information
        - field_mappings_dict: Field name to field ID mappings
        
    Raises:
        ValueError: If template_name not found or neither ID nor name provided
    """
    if template_id:
        # Load by ID (direct)
        print_info(MSG_LOADING_TEMPLATE_BY_ID.format(id=template_id))
        template = fetch_template_details(client, base_url, template_id, headers)
    elif template_name:
        # Load by name (search first, then fetch)
        print_info(MSG_SEARCHING_TEMPLATE_BY_NAME.format(name=template_name))
        templates = fetch_templates(client, base_url, headers)
        # print_templates_list(templates)
        
        template_match = find_template_by_name(templates, template_name)
        if not template_match:
            raise ValueError(MSG_TEMPLATE_NOT_FOUND.format(name=template_name))
        
        template = fetch_template_details(client, base_url, template_match[TEMPLATE_FIELD_ID], headers)
    else:
        raise ValueError("Either template_id or template_name must be provided")
    
    # Extract field mappings and print info
    field_mappings = extract_field_mappings(template)
    print_template_info(template, field_mappings)
    
    return template, field_mappings


def auto_select_template(
    client: httpx.Client,
    base_url: str,
    headers: Dict[str, str]
) -> Tuple[Dict, Dict[str, str]]:
    """
    Automatically select the first available template.
    
    Used as a fallback when no template is specified in configuration.
    Lists all templates and selects the first one.
    
    Args:
        client: HTTP client for API requests
        base_url: API base URL
        headers: Authorization headers
        
    Returns:
        Tuple of (template_dict, field_mappings_dict)
        
    Raises:
        ValueError: If no templates exist in the system
    """
    print_info(MSG_NO_TEMPLATE_IN_ENV)
    templates = fetch_templates(client, base_url, headers)
    print_templates_list(templates)
    
    if not templates:
        raise ValueError(MSG_NO_TEMPLATES_FOUND)
    
    # Select first template
    selected_template = templates[0]
    print_info(MSG_AUTO_SELECT_TEMPLATE.format(name=selected_template.get(TEMPLATE_FIELD_NAME)))
    
    # Load it using the template ID
    return load_template_for_client(
        client, base_url, headers,
        selected_template[TEMPLATE_FIELD_ID], None
    )


# ============================================================================
# METADATA LOADING
# ============================================================================
# Functions to load and filter metadata from HuggingFace repository.

def load_metadata_from_hf(repo_id: str, token: str) -> List[Dict]:
    """
    Load metadata CSV from HuggingFace repository.
    
    Downloads the metadata.csv file and parses it, validating required fields.
    
    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace API token
        
    Returns:
        List of metadata dictionaries, one per valid CSV row
    """
    print_info(MSG_DOWNLOADING_METADATA.format(repo=repo_id))
    
    # Download metadata.csv to current directory
    metadata_path = download_file_from_hf(
        repo_id, FILE_METADATA_CSV, token, Path(".")
    )
    
    return parse_metadata_csv(metadata_path)


def filter_new_files(metadata_list: List[Dict], uploaded_files: Set[str]) -> List[Dict]:
    """
    Filter out already uploaded files from metadata list.
    
    Compares the metadata list against the tracking data of uploaded files
    to identify which files still need to be processed.
    
    Args:
        metadata_list: All files from metadata.csv
        uploaded_files: Set of already uploaded file names from tracking data
        
    Returns:
        List of metadata dicts for files not yet uploaded
    """
    return [
        row for row in metadata_list
        if row[CSV_COLUMN_MAIN_AUDIO_NAME] not in uploaded_files
    ]


# ============================================================================
# MAIN EXECUTION HELPER FUNCTIONS
# ============================================================================
# Functions to print progress and summary information.

def print_configuration(config: Dict) -> None:
    """
    Print configuration summary at startup.
    
    Displays all key configuration values so the user knows what will be processed.
    
    Args:
        config: Configuration dictionary
    """
    print(HEADER_MAIN)
    print()
    template_display = config["template_name"] or config["template_id"]
    print_info(HEADER_CONFIG.format(
        repo=config["hf_repo_id"],
        url=config["api_base_url"],
        template=template_display,
        size=config["batch_size"],
        delay=config["batch_delay_seconds"]
    ))
    print()


def print_metadata_summary(metadata_list: List[Dict], uploaded_files: Set[str], new_files: List[Dict]) -> None:
    """
    Print metadata loading summary.
    
    Shows how many files exist in total, how many are already uploaded,
    and how many new files will be processed.
    
    Args:
        metadata_list: All files from HuggingFace
        uploaded_files: Set of already uploaded files
        new_files: Files to be processed
    """
    print_info(MSG_TOTAL_FILES_HF.format(count=len(metadata_list)))
    print_info(MSG_ALREADY_UPLOADED_COUNT.format(count=len(uploaded_files)))
    print_info(MSG_NEW_FILES_TO_UPLOAD.format(count=len(new_files)))


def print_upload_summary(success_count: int, fail_count: int, total: int, all_uploaded: int) -> None:
    """
    Print final upload summary.
    
    Shows statistics about the upload session and total files uploaded.
    
    Args:
        success_count: Number of successful uploads this session
        fail_count: Number of failed uploads this session
        total: Total number of files attempted this session
        all_uploaded: Total files uploaded across all sessions
    """
    print(HEADER_UPLOAD_COMPLETE)
    print_info(MSG_UPLOAD_SUMMARY.format(success=success_count, total=total))
    print_info(MSG_FAILED_SUMMARY.format(failed=fail_count, total=total))
    print_info(MSG_TOTAL_UPLOADED_ALL_TIME.format(count=all_uploaded))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main() -> None:
    """
    Main execution function - entry point of the script.
    
    This function orchestrates the entire upload process:
    
    1. INITIALIZATION:
       - Load configuration from .env file
       - Validate required settings (including HF_TOKEN)
       - Create HTTP client for API requests
    
    2. TRACKING FILE LOADING:
       - Download tracking file from HuggingFace repo
       - Initialize new tracking data if file doesn't exist
    
    3. AUTHENTICATION:
       - Login to API to get access token
       - Create authorization headers
    
    4. TEMPLATE LOADING:
       - Load metadata template (by ID, name, or auto-select)
       - Extract field mappings for metadata conversion
    
    5. METADATA LOADING:
       - Download metadata.csv from HuggingFace
       - Parse and validate CSV entries
       - Filter out already-uploaded files
    
    6. BATCH PROCESSING:
       - Process files in batches with rate limiting
       - For each file:
         * Download from HuggingFace
         * Upload to API
         * Update tracking data
         * Upload tracking file to HuggingFace
         * Clean up downloaded files
    
    7. SUMMARY:
       - Print statistics (success, failures, total)
       - Close HTTP client connection
    
    The function handles KeyboardInterrupt gracefully, allowing users to
    stop processing with Ctrl+C without data loss.
    """
    # -------------------------------------------------------------------------
    # 1. INITIALIZATION PHASE
    # -------------------------------------------------------------------------
    
    # Load all configuration from environment
    config = load_configuration()
    print_configuration(config)
    
    # Validate we have required configuration
    missing = validate_configuration(config)
    if missing:
        print_error(MSG_MISSING_ENV_VARS.format(vars=", ".join(missing)))
        print_error(MSG_SET_ENV_VARS)
        if "hf_token" in missing:
            print_error(MSG_HF_TOKEN_REQUIRED)
        return
    
    # Initialize components
    workdir = create_directory(Path(DIR_WORKDIR))  # Create working directory
    client = httpx.Client(timeout=HTTP_TIMEOUT_SECONDS)  # Create HTTP client
    
    try:
        # ---------------------------------------------------------------------
        # 2. TRACKING FILE LOADING PHASE
        # ---------------------------------------------------------------------
        
        # Download tracking file from HuggingFace (or create new one)
        tracking_data = download_tracking_file(config["hf_repo_id"], config["hf_token"])

        api = HfApi(token=config["hf_token"])
        remote_file_list = set()
        if config.get("hf_skip_remote_file_list"):
            print_info("Skipping remote file list (HF_SKIP_REMOTE_FILE_LIST is set). Using tracking file only.")
        else:
            print_info("Fetching remote file list from HuggingFace to prevent duplicates...")
            try:
                remote_file_list = set(
                    api.list_repo_files(
                        repo_id=config["hf_repo_id"],
                        repo_type="dataset",
                    )
                )
            except Exception as e:
                print_warn(f"Could not fetch remote file list ({type(e).__name__}). Using tracking file only.")
        
        # ---------------------------------------------------------------------
        # 3. AUTHENTICATION PHASE
        # ---------------------------------------------------------------------
        
        # Login to API and get access token
        access_token = perform_login(
            client,
            config["api_base_url"],
            config["api_username"],
            config["api_password"]
        )
        headers = create_auth_headers(access_token)
        
        # ---------------------------------------------------------------------
        # 4. TEMPLATE LOADING PHASE
        # ---------------------------------------------------------------------
        
        print(HEADER_TEMPLATE_LOADING)
        
        # Load template by ID, name, or auto-select
        if config["template_id"] or config["template_name"]:
            template, field_mappings = load_template_for_client(
                client,
                config["api_base_url"],
                headers,
                config["template_id"],
                config["template_name"]
            )
        else:
            template, field_mappings = auto_select_template(
                client, config["api_base_url"], headers
            )
        
        template_id = template[TEMPLATE_FIELD_ID]
        
        # ---------------------------------------------------------------------
        # 5. METADATA LOADING PHASE
        # ---------------------------------------------------------------------
        
        print(HEADER_LOADING_METADATA)
        
        # Download and parse metadata.csv from HuggingFace
        metadata_list = load_metadata_from_hf(config["hf_repo_id"], config["hf_token"])
        
        # Filter out files that are already uploaded
        uploaded_files = get_all_uploaded_files(tracking_data)
        new_files = filter_new_files(metadata_list, uploaded_files)
        
        print_metadata_summary(metadata_list, uploaded_files, new_files)
        
        # Exit early if nothing to do
        if not new_files:
            print_info(MSG_NO_NEW_FILES)
            return
        
        # ---------------------------------------------------------------------
        # 6. BATCH PROCESSING PHASE
        # ---------------------------------------------------------------------
        
        print(HEADER_PROCESSING_FILES)
        
        # Create a wrapper function that captures all the required parameters
        def process_wrapper(metadata_row: Dict) -> None:
            """Wrapper to process a single file with all required context."""
            process_single_audio_file(
                metadata_row,
                config["hf_repo_id"],
                config["hf_token"],
                workdir,
                client,
                config["api_base_url"],
                headers,
                template_id,
                field_mappings,
                tracking_data,
                remote_file_list
            )
        
        # Process files in batches with rate limiting
        try:
            success_count, fail_count = process_batch_with_delay(
                new_files,
                process_wrapper,
                config["batch_size"],
                config["batch_delay_seconds"]
            )
        except KeyboardInterrupt:
            # User pressed Ctrl+C - stop gracefully
            print(MSG_INTERRUPTED)
            success_count = fail_count = 0
        
        # ---------------------------------------------------------------------
        # 7. SUMMARY PHASE
        # ---------------------------------------------------------------------
        
        # Print final statistics
        all_uploaded = len(get_all_uploaded_files(tracking_data))
        print_upload_summary(success_count, fail_count, len(new_files), all_uploaded)
    
    finally:
        # Always close HTTP client, even if an error occurred
        client.close()


# Entry point when script is run directly
if __name__ == "__main__":
    main()