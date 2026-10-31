"""
Gmail Toolkit for Evo Agent X
-----------------------------

This module provides comprehensive Gmail integration including:
- Message retrieval and reading
- Thread retrieval and reading
- Draft creation, retrieval, update, and sending

Compatible with EvoAgentX tool architecture and follows Gmail API patterns.

IMPORTANT:
Please refer to the setup guide at `docs/tools/Gmail_api_setup.md` 
to configure your Google Cloud Project and credentials before using this toolkit.

"""

import os
import base64
from typing import Dict, Any, List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    raise ImportError(
        "Gmail toolkit requires google-api-python-client and google-auth-oauthlib. "
        "Install with: pip install google-api-python-client google-auth-oauthlib"
    )

from .tool import Tool, Toolkit
from ..core.module import BaseModule
from ..core.logging import logger

# Gmail API scopes
SCOPES = [
    'https://www.googleapis.com/auth/gmail.modify',
    'https://www.googleapis.com/auth/gmail.compose'
]

# Token file name
TOKEN_FILE = 'token.json'
CLIENT_SECRET_FILE = 'client_secret.json'


class GmailBase(BaseModule):
    """
    Base class for Gmail API interactions.
    Handles authentication, client management, and common utilities.
    """
    
    def __init__(self, client_secret_file: str = None, token_file: str = None, **kwargs):
        """
        Initialize the Gmail base.
        
        Args:
            client_secret_file (str, optional): Path to client_secret.json file. 
                If not provided, will try to get from CLIENT_SECRET_FILE or environment variable.
            token_file (str, optional): Path to token.json file. 
                If not provided, will try to get from TOKEN_FILE or environment variable.
            **kwargs: Additional keyword arguments for parent class
        """
        super().__init__(**kwargs)
        
        # Get file paths from parameters or environment variables or defaults
        self.client_secret_file = (
            client_secret_file or 
            os.getenv("GMAIL_CLIENT_SECRET_FILE", CLIENT_SECRET_FILE)
        )
        self.token_file = (
            token_file or 
            os.getenv("GMAIL_TOKEN_FILE", TOKEN_FILE)
        )
        
        # Service will be initialized on first use
        self._service = None
    
    def _get_credentials(self) -> Credentials:
        """
        Get valid user credentials from storage or perform OAuth flow.
        
        Returns:
            Credentials: Valid user credentials
        """
        creds = None
        
        # Load existing token if available
        if os.path.exists(self.token_file):
            try:
                creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)
            except Exception as e:
                logger.warning(f"Error loading token file: {str(e)}")
        
        # If there are no (valid) credentials available, let the user log in
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    logger.warning(f"Error refreshing token: {str(e)}")
                    creds = None
            
            if not creds:
                if not os.path.exists(self.client_secret_file):
                    raise FileNotFoundError(
                        f"Client secret file not found: {self.client_secret_file}. "
                        f"Please download it from Google Cloud Console and place it in the project root."
                    )
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secret_file, SCOPES
                )
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
        
        return creds
    
    def _get_service(self):
        """
        Get Gmail API service instance.
        
        Returns:
            Gmail API service instance
        """
        if self._service is None:
            creds = self._get_credentials()
            self._service = build('gmail', 'v1', credentials=creds)
        return self._service
    
    def _format_message(self, message: Dict[str, Any], include_attachments: bool = False) -> Dict[str, Any]:
        """
        Format a Gmail message for consistent output.
        
        Args:
            message: Gmail message object from API
            include_attachments: Whether to include attachment information
            
        Returns:
            dict: Formatted message data
        """
        payload = message.get('payload', {})
        headers = payload.get('headers', [])
        
        # Extract common headers
        header_dict = {h['name'].lower(): h['value'] for h in headers}
        
        # Get message body
        body = ""
        
        def decode_body(data_str: str) -> str:
            """Helper function to decode base64 body data."""
            if not data_str:
                return ""
            try:
                # Add padding if needed
                padding = 4 - len(data_str) % 4
                if padding != 4:
                    data_str += '=' * padding
                return base64.urlsafe_b64decode(data_str).decode('utf-8')
            except Exception as e:
                logger.warning(f"Error decoding body: {str(e)}")
                return ""
        
        def extract_body_from_parts(parts: List[Dict]) -> str:
            """Recursively extract body from message parts."""
            html_body = ""  # Store HTML as fallback
            for part in parts:
                mime_type = part.get('mimeType', '')
                if mime_type == 'text/plain':
                    data = part.get('body', {}).get('data', '')
                    if data:
                        return decode_body(data)
                elif mime_type == 'text/html':
                    # Store HTML as fallback if plain text not found
                    data = part.get('body', {}).get('data', '')
                    if data and not html_body:
                        html_body = decode_body(data)
                elif 'parts' in part:
                    # Recursively check nested parts
                    nested_body = extract_body_from_parts(part['parts'])
                    if nested_body:
                        return nested_body
            return html_body  # Return HTML if no plain text found
        
        if 'parts' in payload:
            body = extract_body_from_parts(payload['parts'])
        else:
            # Single part message
            mime_type = payload.get('mimeType', '')
            if mime_type in ['text/plain', 'text/html']:
                data = payload.get('body', {}).get('data', '')
                if data:
                    body = decode_body(data)
        
        # Extract attachment information
        attachments_info = []
        if include_attachments:
            parts = payload.get('parts', [])
            for part in parts:
                if part.get('filename'):
                    attachments_info.append({
                        "filename": part.get('filename'),
                        "mime_type": part.get('mimeType'),
                        "attachment_id": part.get('body', {}).get('attachmentId'),
                        "size": part.get('body', {}).get('size', 0)
                    })
        
        formatted = {
            "id": message.get('id'),
            "thread_id": message.get('threadId'),
            "subject": header_dict.get('subject', ''),
            "from": header_dict.get('from', ''),
            "to": header_dict.get('to', ''),
            "cc": header_dict.get('cc', ''),
            "bcc": header_dict.get('bcc', ''),
            "date": header_dict.get('date', ''),
            "snippet": message.get('snippet', ''),
            "body": body,
            "label_ids": message.get('labelIds', [])
        }
        
        if include_attachments:
            formatted["attachments_count"] = len(attachments_info)
            formatted["attachments"] = attachments_info
        
        return formatted

    def _ensure_label(self, label_name: str) -> str:
        """
        Ensure a label exists in the user's mailbox. For system labels (e.g., INBOX, STARRED)
        return the system label id/name. For custom labels, return existing id or create one.

        Returns:
            label id (string)
        """
        if not label_name:
            return ""

        # Normalize common system labels to their API ids
        system_labels = {
            'INBOX', 'SENT', 'DRAFT', 'TRASH', 'SPAM', 'IMPORTANT', 'STARRED', 'UNREAD'
        }

        # If it's a known system label, return uppercase id
        if label_name.upper() in system_labels:
            return label_name.upper()

        try:
            service = self._get_service()

            # List existing labels to find by name
            labels_res = service.users().labels().list(userId='me').execute()
            for lab in labels_res.get('labels', []):
                if lab.get('name') == label_name:
                    return lab.get('id')

            # Create the label if not found
            create_body = {
                'name': label_name,
                'labelListVisibility': 'labelShow',
                'messageListVisibility': 'show'
            }
            new_label = service.users().labels().create(userId='me', body=create_body).execute()
            return new_label.get('id')
        except Exception as e:
            logger.warning(f"Could not ensure label '{label_name}': {e}")
            return label_name

    def _modify_message_labels(self, message_id: str, add_labels: list = None, remove_labels: list = None) -> Dict[str, Any]:
        """
        Modify labels for a single message by id. Label names will be resolved to ids,
        creating custom labels when necessary.
        """
        try:
            service = self._get_service()

            add_ids = []
            remove_ids = []

            if add_labels:
                for lname in add_labels:
                    lid = self._ensure_label(lname)
                    if lid:
                        add_ids.append(lid)

            if remove_labels:
                for lname in remove_labels:
                    lid = self._ensure_label(lname)
                    if lid:
                        remove_ids.append(lid)

            body = {}
            if add_ids:
                body['addLabelIds'] = add_ids
            if remove_ids:
                body['removeLabelIds'] = remove_ids

            if not body:
                return {"success": False, "error": "No labels provided to add or remove."}

            res = service.users().messages().modify(userId='me', id=message_id, body=body).execute()
            return {"success": True, "result": res}
        except HttpError as e:
            logger.error(f"Error modifying labels for message {message_id}: {str(e)}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected error modifying labels for {message_id}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _create_message(self, to: str, subject: str, body: str, 
                       cc: str = None, bcc: str = None, 
                       is_html: bool = False, attachments: list = None) -> Dict[str, str]:
        """
        Create a message for sending or drafting with optional attachments.
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            cc: CC email address (optional)
            bcc: BCC email address (optional)
            is_html: Whether body is HTML (default: False)
            attachments: List of file paths to attach (optional)
            
        Returns:
            dict: Message object ready for API
        """
        if attachments:
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject
            if cc:
                message['cc'] = cc
            if bcc:
                message['bcc'] = bcc
            
            # Add body
            message.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            # Add attachments
            for file_path in attachments:
                if not os.path.exists(file_path):
                    logger.warning(f"Attachment file not found: {file_path}")
                    continue
                
                try:
                    filename = os.path.basename(file_path)
                    with open(file_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                    encoders.encode_base64(part)
                    part.add_header('Content-Disposition', 'attachment', filename=filename)
                    message.attach(part)
                except Exception as e:
                    logger.warning(f"Error attaching file {file_path}: {str(e)}")
                    continue
        else:
            message = MIMEText(body, 'html' if is_html else 'plain')
            message['to'] = to
            message['subject'] = subject
            if cc:
                message['cc'] = cc
            if bcc:
                message['bcc'] = bcc
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
        return {'raw': raw_message}
    
    def _download_attachments(self, message_id: str, download_dir: str = None) -> Dict[str, Any]:
        """
        Download all attachments from a Gmail message with unique filename handling.
        Appends timestamp to filenames to prevent silent overwrites (format: filename_YYYYMMDD_HHMMSS_counter.ext)
        
        Args:
            message_id: ID of the message to download attachments from
            download_dir: Directory to save attachments (optional, defaults to current dir)
            
        Returns:
            Dictionary with attachment information and file paths
        """
        if download_dir is None:
            download_dir = os.getcwd()
        
        # Create download directory if it doesn't exist
        os.makedirs(download_dir, exist_ok=True)
        
        try:
            import time
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            
            service = self._get_service()
            message = service.users().messages().get(
                userId='me',
                id=message_id,
                format='full'
            ).execute()
            
            payload = message.get('payload', {})
            parts = payload.get('parts', [])
            
            attachments_info = []
            attachment_counter = 0
            
            for part in parts:
                if part.get('filename'):
                    try:
                        filename = part.get('filename')
                        # Sanitize filename and generate unique name
                        name_parts = os.path.splitext(filename)
                        base_name = name_parts[0]
                        ext = name_parts[1] if len(name_parts) > 1 else ''
                        
                        # Create unique filename: name_YYYYMMDD_HHMMSS_counter.ext
                        unique_filename = f"{base_name}_attachment_{timestamp}_{attachment_counter}{ext}"
                        attachment_counter += 1
                        
                        attachment_id = part.get('body', {}).get('attachmentId')
                        
                        if attachment_id:
                            # Download the attachment
                            attachment = service.users().messages().attachments().get(
                                userId='me',
                                messageId=message_id,
                                id=attachment_id
                            ).execute()
                            
                            data = attachment.get('data')
                            if data:
                                file_data = base64.urlsafe_b64decode(data)
                                file_path = os.path.join(download_dir, unique_filename)
                                
                                with open(file_path, 'wb') as f:
                                    f.write(file_data)
                                
                                attachments_info.append({
                                    "original_filename": filename,
                                    "saved_filename": unique_filename,
                                    "attachment_id": attachment_id,
                                    "size": len(file_data),
                                    "file_path": file_path
                                })
                    except Exception as e:
                        logger.warning(f"Error downloading attachment {part.get('filename')}: {str(e)}")
                        continue
            
            return {
                "success": True,
                "attachments_count": len(attachments_info),
                "download_dir": download_dir,
                "attachments": attachments_info
            }
            
        except HttpError as e:
            logger.error(f"Error downloading attachments: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to download attachments: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error downloading attachments: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _fetch_messages(self, query: str = None, label_ids: list = None, max_results: int = 10, 
                       include_attachments: bool = False, download_attachments: bool = False, 
                       download_dir: str = None) -> Dict[str, Any]:
        """
        Shared helper to fetch and format messages from Gmail.
        Used by both read_messages and search_messages tools.
        
        Args:
            query: Gmail search query (optional)
            label_ids: List of label IDs to filter by (optional)
            max_results: Maximum number of messages to retrieve
            include_attachments: Whether to include attachment info in response
            download_attachments: Whether to download attachments to disk
            download_dir: Directory to download attachments to (used if download_attachments=True)
            
        Returns:
            Dictionary with formatted messages
        """
        try:
            service = self._get_service()
            query_str = query or ""
            
            # Auto-enable include_attachments if we're downloading
            if download_attachments:
                include_attachments = True
            
            # Get message list
            results = service.users().messages().list(
                userId='me',
                q=query_str,
                labelIds=label_ids,
                maxResults=min(max_results, 500)
            ).execute()
            
            messages = results.get('messages', [])
            
            if not messages:
                return {
                    "success": True,
                    "messages_count": 0,
                    "messages": []
                }
            
            # Fetch full message details
            formatted_messages = []
            for msg in messages:
                try:
                    message = service.users().messages().get(
                        userId='me',
                        id=msg['id'],
                        format='full'
                    ).execute()
                    formatted_msg = self._format_message(message, include_attachments=include_attachments)
                    
                    # Download attachments if requested
                    if download_attachments and formatted_msg.get('attachments_count', 0) > 0:
                        attach_result = self._download_attachments(msg['id'], download_dir)
                        if attach_result.get('success'):
                            formatted_msg['downloaded_attachments'] = attach_result['attachments']
                    
                    formatted_messages.append(formatted_msg)
                except HttpError as e:
                    logger.warning(f"Error fetching message {msg['id']}: {str(e)}")
                    continue
            
            return {
                "success": True,
                "messages_count": len(formatted_messages),
                "total_available": len(formatted_messages),
                "display_limit": 10,
                "messages": formatted_messages[:10]
            }
            
        except HttpError as e:
            logger.error(f"Error fetching messages: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to fetch messages: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error fetching messages: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def _fetch_threads(self, query: str = None, label_ids: list = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Shared helper to fetch and format Gmail threads.
        Used by read_threads tool to centralize thread-fetching logic.
        
        Args:
            query: Gmail search query (optional)
            label_ids: List of label IDs to filter by (optional)
            max_results: Maximum number of threads to retrieve
            
        Returns:
            Dictionary with formatted threads
        """
        try:
            service = self._get_service()
            query_str = query or ""
            
            # Get thread list
            results = service.users().threads().list(
                userId='me',
                q=query_str,
                labelIds=label_ids,
                maxResults=min(max_results, 500)
            ).execute()
            
            threads = results.get('threads', [])
            
            if not threads:
                return {
                    "success": True,
                    "threads_count": 0,
                    "threads": []
                }
            
            # Fetch full thread details
            formatted_threads = []
            for thread in threads:
                try:
                    thread_data = service.users().threads().get(
                        userId='me',
                        id=thread['id'],
                        format='full'
                    ).execute()
                    
                    # Format all messages in the thread
                    messages = []
                    for message in thread_data.get('messages', []):
                        messages.append(self._format_message(message))
                    
                    formatted_threads.append({
                        "id": thread_data.get('id'),
                        "history_id": thread_data.get('historyId'),
                        "messages_count": len(messages),
                        "messages": messages
                    })
                except HttpError as e:
                    logger.warning(f"Error fetching thread {thread['id']}: {str(e)}")
                    continue
            
            return {
                "success": True,
                "threads_count": len(formatted_threads),
                "total_available": len(formatted_threads),
                "display_limit": 10,
                "threads": formatted_threads[:10]
            }
            
        except HttpError as e:
            logger.error(f"Error fetching threads: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to fetch threads: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error fetching threads: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class ReadMessagesTool(Tool):
    """Read Gmail messages with optional filtering and attachment download."""
    
    name: str = "read_messages"
    description: str = (
        "Read Gmail messages with optional filtering by query, label, or max results. "
        "Can optionally include attachment information and/or download attachments. "
        "Returns a list of messages with their details including subject, from, to, date, snippet, and body."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "Gmail search query (e.g., 'from:example@gmail.com', 'subject:meeting', 'is:unread'). Optional."
        },
        "label_ids": {
            "type": "array",
            "description": "List of label IDs to filter by (e.g., ['INBOX', 'UNREAD']). Optional.",
            "items": {"type": "string"}
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of messages to retrieve (default: 10, max: 500)"
        },
        "include_attachments": {
            "type": "boolean",
            "description": "Whether to include attachment information in the response (default: false)"
        },
        "download_attachments": {
            "type": "boolean",
            "description": "Whether to download attachments to disk (default: false)"
        },
        "download_dir": {
            "type": "string",
            "description": "Directory to save downloaded attachments (optional, defaults to current directory)"
        }
    }
    required: List[str] = []
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, query: str = None, label_ids: list = None, max_results: int = 10,
                 include_attachments: bool = False, download_attachments: bool = False,
                 download_dir: str = None) -> Dict[str, Any]:
        """
        Read Gmail messages with optional attachment handling.
        
        Args:
            query: Gmail search query
            label_ids: List of label IDs to filter by
            max_results: Maximum number of messages to retrieve
            include_attachments: Whether to include attachment information
            download_attachments: Whether to download attachments
            download_dir: Directory to save attachments
            
        Returns:
            Dictionary with message results
        """
        return self.gmail_base._fetch_messages(
            query, label_ids, max_results, 
            include_attachments=include_attachments,
            download_attachments=download_attachments,
            download_dir=download_dir
        )


class SearchMessagesTool(Tool):
    """Search Gmail messages using Gmail's search syntax."""
    
    name: str = "search_messages"
    description: str = (
        "Search Gmail messages using Gmail's search syntax (e.g., "
        "'from:example@gmail.com subject:\"weekly report\" is:unread newer_than:7d'). "
        "Returns a list of matching messages with details including subject, from, to, date, snippet, and body."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "keyword": {
            "type": "string",
            "description": "Gmail search query / keyword expression using Gmail's search syntax."
        },
        "label_ids": {
            "type": "array",
            "description": "Optional list of label IDs to filter by (e.g., ['INBOX', 'UNREAD']).",
            "items": {"type": "string"}
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of messages to retrieve (default: 10, max: 500)"
        },
        "include_attachments": {
            "type": "boolean",
            "description": "Whether to include attachment information in the response (default: false)"
        }
    }
    required: List[str] = ["keyword"]
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, keyword: str, label_ids: list = None, max_results: int = 10, include_attachments: bool = False) -> Dict[str, Any]:
        """
        Search Gmail messages.
        
        Args:
            keyword: Gmail search query / keyword expression
            label_ids: Optional list of label IDs to filter by
            max_results: Maximum number of messages to retrieve
            include_attachments: Whether to include attachment information
            
        Returns:
            Dictionary with search results
        """
        result = self.gmail_base._fetch_messages(keyword, label_ids, max_results, include_attachments=include_attachments)
        # Add query field to maintain API consistency
        if result.get('success'):
            result['query'] = keyword or ""
        return result


class ReadThreadsTool(Tool):
    """Read Gmail threads with optional filtering."""
    
    name: str = "read_threads"
    description: str = (
        "Read Gmail threads (conversations) with optional filtering by query or label. "
        "Returns a list of threads with all messages in each thread."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "query": {
            "type": "string",
            "description": "Gmail search query (e.g., 'from:example@gmail.com', 'subject:meeting'). Optional."
        },
        "label_ids": {
            "type": "array",
            "description": "List of label IDs to filter by (e.g., ['INBOX', 'UNREAD']). Optional.",
            "items": {"type": "string"}
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of threads to retrieve (default: 10, max: 500)"
        }
    }
    required: List[str] = []
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, query: str = None, label_ids: list = None, max_results: int = 10) -> Dict[str, Any]:
        """
        Read Gmail threads.
        
        Args:
            query: Gmail search query
            label_ids: List of label IDs to filter by
            max_results: Maximum number of threads to retrieve
            
        Returns:
            Dictionary with thread results
        """
        return self.gmail_base._fetch_threads(query, label_ids, max_results)


class TrashMessagesTool(Tool):
    """Move Gmail messages to Trash by IDs or search query."""
    
    name: str = "trash_messages"
    description: str = (
        "Move Gmail messages to Trash either by providing explicit message IDs or by using "
        "a Gmail search query (Gmail search syntax). This does NOT permanently delete messages; "
        "they are moved to the Trash label."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "message_ids": {
            "type": "array",
            "description": "Optional list of message IDs to move to Trash (as returned by read/search tools).",
            "items": {"type": "string"}
        },
        "query": {
            "type": "string",
            "description": (
                "Optional Gmail search query to select messages to trash "
                "(e.g., 'subject:\"Weekly Report\" is:read older_than:30d')."
            )
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of messages to match when using query (default: 50, max: 500)."
        }
    }
    # No unconditional required fields; at least one of message_ids or query must be provided
    required: List[str] = []
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(
        self,
        message_ids: list = None,
        query: str = None,
        max_results: int = 50
    ) -> Dict[str, Any]:
        """
        Move Gmail messages to Trash.
        
        Args:
            message_ids: Optional list of explicit message IDs to trash.
            query: Optional Gmail search query to select messages to trash.
            max_results: Maximum number of messages to match when using query.
            
        Returns:
            Dictionary with trash operation results.
        """
        try:
            service = self.gmail_base._get_service()
            
            # Normalize inputs
            target_ids: list = list(message_ids or [])
            
            # If no explicit IDs, resolve via search query
            if not target_ids and query:
                results = service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=min(max_results, 500)
                ).execute()
                msgs = results.get('messages', [])
                target_ids = [m['id'] for m in msgs]
            
            if not target_ids:
                return {
                    "success": False,
                    "error": "No messages to trash. Provide message_ids or a query that matches messages."
                }
            
            trashed_ids: list = []
            failed_ids: list = []
            
            for msg_id in target_ids:
                try:
                    service.users().messages().trash(
                        userId='me',
                        id=msg_id
                    ).execute()
                    trashed_ids.append(msg_id)
                except HttpError as e:
                    logger.warning(f"Error trashing message {msg_id}: {str(e)}")
                    failed_ids.append({"id": msg_id, "error": str(e)})
                except Exception as e:
                    logger.warning(f"Unexpected error trashing message {msg_id}: {str(e)}")
                    failed_ids.append({"id": msg_id, "error": str(e)})
            
            return {
                "success": True,
                "requested_ids": target_ids,
                "trashed_ids": trashed_ids,
                "failed_ids": failed_ids,
                "message": f"Moved {len(trashed_ids)} messages to Trash; {len(failed_ids)} failed."
            }
        
        except HttpError as e:
            logger.error(f"Error trashing messages: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to trash messages: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error trashing messages: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class ModifyLabelsTool(Tool):
    """Add or remove labels on messages (creates custom labels if necessary)."""

    name: str = "modify_labels"
    description: str = (
        "Add or remove labels on messages. Provide message_ids or a query to select messages. "
        "Label names will be resolved to system ids or created if custom."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "message_ids": {"type": "array", "description": "List of message IDs to modify", "items": {"type": "string"}},
        "query": {"type": "string", "description": "Gmail query to select messages (optional)"},
        "add_labels": {
            "type": "array",
            "description": "List of label names to add (e.g., ['STARRED','MyLabel'])",
            "items": {"type": "string"}
        },
        "remove_labels": {"type": "array", "description": "List of label names to remove", "items": {"type": "string"}},
        "max_results": {"type": "integer", "description": "Max messages when using query (default 50)"}
    }
    required: List[str] = []

    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base

    def __call__(self, message_ids: list = None, query: str = None, add_labels: list = None, remove_labels: list = None, max_results: int = 50) -> Dict[str, Any]:
        try:
            service = self.gmail_base._get_service()

            targets = list(message_ids or [])

            if not targets and query:
                results = service.users().messages().list(userId='me', q=query, maxResults=min(max_results, 500)).execute()
                msgs = results.get('messages', [])
                targets = [m['id'] for m in msgs]

            if not targets:
                return {"success": False, "error": "No target messages. Provide message_ids or a query."}

            modified = []
            failed = []
            for mid in targets:
                res = self.gmail_base._modify_message_labels(mid, add_labels=add_labels, remove_labels=remove_labels)
                if res.get('success'):
                    modified.append({"id": mid, "result": res.get('result')})
                else:
                    failed.append({"id": mid, "error": res.get('error')})

            return {"success": True, "modified": modified, "failed": failed}
        except Exception as e:
            logger.error(f"Unexpected error modifying labels: {str(e)}")
            return {"success": False, "error": str(e)}


class CreateDraftTool(Tool):
    """Create a Gmail draft with optional attachments."""
    
    name: str = "create_draft"
    description: str = (
        "Create a Gmail draft message with optional file attachments. "
        "The draft can be sent later using the send_draft tool."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "to": {
            "type": "string",
            "description": "Recipient email address"
        },
        "subject": {
            "type": "string",
            "description": "Email subject"
        },
        "body": {
            "type": "string",
            "description": "Email body text"
        },
        "cc": {
            "type": "string",
            "description": "CC email address (optional)"
        },
        "bcc": {
            "type": "string",
            "description": "BCC email address (optional)"
        },
        "is_html": {
            "type": "boolean",
            "description": "Whether the body is HTML format (default: false)"
        },
        "attachments": {
            "type": "array",
            "description": "List of file paths to attach to the draft (optional, e.g., ['/path/to/file1.pdf', '/path/to/file2.txt'])",
            "items": {"type": "string"}
        }
    }
    required: List[str] = ["to", "subject", "body"]
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, to: str, subject: str, body: str, 
                 cc: str = None, bcc: str = None, is_html: bool = False,
                 attachments: list = None) -> Dict[str, Any]:
        """
        Create a Gmail draft with optional attachments.
        
        Args:
            to: Recipient email address
            subject: Email subject
            body: Email body
            cc: CC email address (optional)
            bcc: BCC email address (optional)
            is_html: Whether body is HTML (default: False)
            attachments: List of file paths to attach (optional)
            
        Returns:
            Dictionary with draft creation result
        """
        try:
            service = self.gmail_base._get_service()
            
            # Create message with optional attachments
            message = self.gmail_base._create_message(to, subject, body, cc, bcc, is_html, attachments)
            
            # Create draft
            draft = service.users().drafts().create(
                userId='me',
                body={'message': message}
            ).execute()
            
            result = {
                "success": True,
                "draft_id": draft.get('id'),
                "message_id": draft.get('message', {}).get('id'),
                "message": "Draft created successfully"
            }
            
            # Include attachment info if attachments were added
            if attachments:
                result["attachments_count"] = len(attachments)
                result["attachments"] = attachments
            
            return result
            
        except HttpError as e:
            logger.error(f"Error creating draft: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to create draft: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error creating draft: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class ListDraftsTool(Tool):
    """List all Gmail drafts."""
    
    name: str = "list_drafts"
    description: str = (
        "List all Gmail drafts. Returns a list of draft IDs and basic information."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "max_results": {
            "type": "integer",
            "description": "Maximum number of drafts to retrieve (default: 10, max: 500)"
        }
    }
    required: List[str] = []
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, max_results: int = 10) -> Dict[str, Any]:
        """
        List all Gmail drafts.
        
        Args:
            max_results: Maximum number of drafts to retrieve
            
        Returns:
            Dictionary with list of drafts
        """
        try:
            service = self.gmail_base._get_service()
            
            # List drafts
            results = service.users().drafts().list(
                userId='me',
                maxResults=min(max_results, 500)
            ).execute()
            
            drafts = results.get('drafts', [])
            
            # Get basic info for each draft
            draft_list = []
            for draft in drafts:
                try:
                    draft_data = service.users().drafts().get(
                        userId='me',
                        id=draft['id'],
                        format='full'
                    ).execute()
                    
                    message = draft_data.get('message', {})
                    headers = message.get('payload', {}).get('headers', [])
                    header_dict = {h['name'].lower(): h['value'] for h in headers}
                    
                    draft_list.append({
                        "draft_id": draft['id'],
                        "message_id": message.get('id'),
                        "thread_id": message.get('threadId'),
                        "subject": header_dict.get('subject', ''),
                        "to": header_dict.get('to', ''),
                        "snippet": message.get('snippet', '')
                    })
                except HttpError as e:
                    logger.warning(f"Error fetching draft {draft['id']}: {str(e)}")
                    continue
            
            return {
                "success": True,
                "drafts_count": len(draft_list),
                "total_available": len(draft_list),
                "display_limit": 10,
                "drafts": draft_list[:10]
            }
            
        except HttpError as e:
            logger.error(f"Error listing drafts: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to list drafts: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error listing drafts: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class SearchDraftTool(Tool):
    """Search and retrieve drafts by ID or keyword."""

    name: str = "search_drafts"
    description: str = (
        "Unified tool to retrieve or search Gmail drafts. "
        "Provide `draft_id` to fetch a specific draft directly (fast), or provide `keyword` to search "
        "across draft subject/sender/recipient/snippet/body for matches (scans multiple drafts). "
        "Either draft_id or keyword required."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "draft_id": {"type": "string", "description": "Draft ID to retrieve directly (optional; if provided, returns that draft immediately)"},
        "keyword": {"type": "string", "description": "Keyword to search in drafts (optional; scans multiple drafts if provided)"},
        "max_results": {"type": "integer", "description": "Max drafts to scan when using keyword search (default 50)"}
    }
    required: List[str] = []

    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base

    def __call__(self, draft_id: str = None, keyword: str = None, max_results: int = 50) -> Dict[str, Any]:
        """
        Search for or retrieve drafts.
        
        Args:
            draft_id: Draft ID to retrieve directly (optional)
            keyword: Keyword to search in drafts (optional)
            max_results: Max drafts to scan when searching (default 50)
            
        Returns:
            Dictionary with draft(s)
        """
        try:
            service = self.gmail_base._get_service()

            if draft_id:
                draft = service.users().drafts().get(userId='me', id=draft_id, format='full').execute()
                message = draft.get('message', {})
                formatted = self.gmail_base._format_message(message, include_attachments=True)
                return {"success": True, "drafts_count": 1, "drafts": [formatted]}

            if not keyword:
                return {"success": False, "error": "Provide draft_id or keyword to search."}

            # List drafts and scan
            results = service.users().drafts().list(userId='me', maxResults=min(max_results, 500)).execute()
            drafts = results.get('drafts', [])
            matches = []
            for d in drafts:
                try:
                    draft_data = service.users().drafts().get(userId='me', id=d['id'], format='full').execute()
                    msg = draft_data.get('message', {})
                    formatted = self.gmail_base._format_message(msg, include_attachments=True)

                    # simple case-insensitive search in subject, from, to, snippet, body
                    k = keyword.lower()
                    if any(k in (str(formatted.get(k2, '')).lower()) for k2 in ("subject", "from", "to", "snippet")) \
                       or (formatted.get('body') and k in formatted.get('body', '').lower()):
                        matches.append(formatted)
                except HttpError as e:
                    logger.warning(f"Error fetching draft {d.get('id')}: {str(e)}")
                    continue

            return {"success": True, "query": keyword, "drafts_count": len(matches), "total_available": len(matches), "display_limit": 10, "drafts": matches[:10]}

        except HttpError as e:
            logger.error(f"Error searching/retrieving drafts: {str(e)}")
            return {"success": False, "error": f"Failed to search/retrieve drafts: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error searching/retrieving drafts: {str(e)}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}


class DeleteDraftTool(Tool):
    """Delete a draft by ID."""

    name: str = "delete_draft"
    description: str = "Delete a Gmail draft by its ID. Returns success/failure." 
    inputs: Dict[str, Dict[str, str]] = {
        "draft_id": {"type": "string", "description": "ID of the draft to delete"}
    }
    required: List[str] = ["draft_id"]

    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base

    def __call__(self, draft_id: str) -> Dict[str, Any]:
        try:
            service = self.gmail_base._get_service()
            service.users().drafts().delete(userId='me', id=draft_id).execute()
            return {"success": True, "draft_id": draft_id, "message": "Draft deleted successfully"}
        except HttpError as e:
            logger.error(f"Error deleting draft {draft_id}: {str(e)}")
            return {"success": False, "error": f"Failed to delete draft: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error deleting draft {draft_id}: {str(e)}")
            return {"success": False, "error": f"Unexpected error: {str(e)}"}


class UpdateDraftTool(Tool):
    """Update an existing Gmail draft with optional attachments."""
    
    name: str = "update_draft"
    description: str = (
        "Update an existing Gmail draft. You can update the recipient, subject, body, or add attachments."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "draft_id": {
            "type": "string",
            "description": "The ID of the draft to update"
        },
        "to": {
            "type": "string",
            "description": "Recipient email address (optional, only if updating)"
        },
        "subject": {
            "type": "string",
            "description": "Email subject (optional, only if updating)"
        },
        "body": {
            "type": "string",
            "description": "Email body (optional, only if updating)"
        },
        "cc": {
            "type": "string",
            "description": "CC email address (optional)"
        },
        "bcc": {
            "type": "string",
            "description": "BCC email address (optional)"
        },
        "is_html": {
            "type": "boolean",
            "description": "Whether the body is HTML format (default: false)"
        },
        "attachments": {
            "type": "array",
            "description": "List of file paths to attach to the draft (optional)",
            "items": {"type": "string"}
        }
    }
    required: List[str] = ["draft_id"]
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, draft_id: str, to: str = None, subject: str = None, 
                 body: str = None, cc: str = None, bcc: str = None, 
                 is_html: bool = False, attachments: list = None) -> Dict[str, Any]:
        """
        Update a Gmail draft with optional attachments.
        
        Args:
            draft_id: The ID of the draft to update
            to: Recipient email address (optional)
            subject: Email subject (optional)
            body: Email body (optional)
            cc: CC email address (optional)
            bcc: BCC email address (optional)
            is_html: Whether body is HTML (default: False)
            attachments: List of file paths to attach (optional)
            
        Returns:
            Dictionary with update result
        """
        try:
            service = self.gmail_base._get_service()
            
            # Get existing draft
            existing_draft = service.users().drafts().get(
                userId='me',
                id=draft_id,
                format='full'
            ).execute()
            
            existing_message = existing_draft.get('message', {})
            existing_payload = existing_message.get('payload', {})
            existing_headers = existing_payload.get('headers', [])
            header_dict = {h['name'].lower(): h['value'] for h in existing_headers}
            
            # Use existing values if not provided
            final_to = to or header_dict.get('to', '')
            final_subject = subject or header_dict.get('subject', '')
            final_cc = cc or header_dict.get('cc')
            final_bcc = bcc or header_dict.get('bcc')
            
            # Get existing body if not provided
            if body is None:
                # Extract existing body
                if 'parts' in existing_payload:
                    for part in existing_payload['parts']:
                        if part.get('mimeType') in ['text/plain', 'text/html']:
                            data = part.get('body', {}).get('data', '')
                            if data:
                                final_body = base64.urlsafe_b64decode(data).decode('utf-8')
                                is_html = part.get('mimeType') == 'text/html'
                                break
                    else:
                        final_body = ""
                else:
                    if existing_payload.get('mimeType') in ['text/plain', 'text/html']:
                        data = existing_payload.get('body', {}).get('data', '')
                        final_body = base64.urlsafe_b64decode(data).decode('utf-8') if data else ""
                        is_html = existing_payload.get('mimeType') == 'text/html'
                    else:
                        final_body = ""
            else:
                final_body = body
            
            # Create updated message with optional attachments
            message = self.gmail_base._create_message(
                final_to, final_subject, final_body, final_cc, final_bcc, is_html, attachments
            )
            
            # Update draft
            updated_draft = service.users().drafts().update(
                userId='me',
                id=draft_id,
                body={'message': message}
            ).execute()
            
            result = {
                "success": True,
                "draft_id": updated_draft.get('id'),
                "message_id": updated_draft.get('message', {}).get('id'),
                "message": "Draft updated successfully"
            }
            
            if attachments:
                result["attachments_count"] = len(attachments)
                result["attachments"] = attachments
            
            return result
            
        except HttpError as e:
            logger.error(f"Error updating draft: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to update draft: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error updating draft: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class SendDraftTool(Tool):
    """Send a Gmail draft."""
    
    name: str = "send_draft"
    description: str = (
        "Send a Gmail draft by its ID. The draft will be sent as an email and removed from drafts."
    )
    inputs: Dict[str, Dict[str, str]] = {
        "draft_id": {
            "type": "string",
            "description": "The ID of the draft to send"
        }
    }
    required: List[str] = ["draft_id"]
    
    def __init__(self, gmail_base: GmailBase):
        super().__init__()
        self.gmail_base = gmail_base
    
    def __call__(self, draft_id: str) -> Dict[str, Any]:
        """
        Send a Gmail draft.
        
        Args:
            draft_id: The ID of the draft to send
            
        Returns:
            Dictionary with send result
        """
        try:
            service = self.gmail_base._get_service()
            '''
            # Get draft
            draft = service.users().drafts().get(
                userId='me',
                id=draft_id,
                format='full'
            ).execute()'''
            
            #message = draft.get('message', {})
            
            # Send the draft
            sent_message = service.users().drafts().send(
                userId='me',
                body={'id': draft_id}
            ).execute()
            
            return {
                "success": True,
                "message_id": sent_message.get('id'),
                "thread_id": sent_message.get('threadId'),
                "label_ids": sent_message.get('labelIds', []),
                "message": "Draft sent successfully"
            }
            
        except HttpError as e:
            logger.error(f"Error sending draft: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to send draft: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Unexpected error sending draft: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }


class GmailToolkit(Toolkit):
    """
    Complete Gmail toolkit containing all available tools.
    """
    
    def __init__(self, client_secret_file: str = None, token_file: str = None, 
                 name: str = "GmailToolkit"):
        """
        Initialize the Gmail toolkit.
        
        Args:
            client_secret_file (str, optional): Path to client_secret.json file. 
                If not provided, will try to get from CLIENT_SECRET_FILE or environment variable.
            token_file (str, optional): Path to token.json file. 
                If not provided, will try to get from TOKEN_FILE or environment variable.
            name (str): Toolkit name
        """
        # Create shared Gmail base instance
        gmail_base = GmailBase(client_secret_file=client_secret_file, token_file=token_file)
        
        # Create all tools with shared base
        tools = [
            ReadMessagesTool(gmail_base=gmail_base),
            SearchMessagesTool(gmail_base=gmail_base),
            ReadThreadsTool(gmail_base=gmail_base),
            ListDraftsTool(gmail_base=gmail_base),
            CreateDraftTool(gmail_base=gmail_base),
            SearchDraftTool(gmail_base=gmail_base),
            UpdateDraftTool(gmail_base=gmail_base),
            SendDraftTool(gmail_base=gmail_base),
            DeleteDraftTool(gmail_base=gmail_base),
            TrashMessagesTool(gmail_base=gmail_base),
            ModifyLabelsTool(gmail_base=gmail_base)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store base instance for access
        self.gmail_base = gmail_base

