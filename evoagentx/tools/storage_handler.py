import os
import shutil
import mimetypes
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
from abc import abstractmethod

from .storage_base import StorageBase
from ..core.logging import logger


class FileStorageHandler(StorageBase):
    """
    Reference implementation showing all available _raw_xxx methods.
    This class serves as a template for developers creating new storage handlers.
    Concrete handlers only need to implement the _raw_xxx methods they need.
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize the storage handler.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
    
    # ____________________ How to use it ____________________ #
    def create(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        return super().save(file_path, content, **kwargs)
    
    def read(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return super().read(file_path, **kwargs)
    
    def list(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        return super().list(path, max_depth, include_hidden)
    
    def delete(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return super().delete(file_path, **kwargs)
    
    def move(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return super().move(source, destination, **kwargs)
    
    def copy(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return super().copy(source, destination, **kwargs)
    
    def create_directory(self, path: str, **kwargs) -> Dict[str, Any]:
        return super().create_directory(path, **kwargs)
    
    
    
    # ____________________ Required Methods ____________________ #
    @abstractmethod
    def _initialize_storage(self):
        """Initialize storage - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _delete_raw(self, path: str) -> bool:
        """Delete file or directory - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def _list_raw(self, path: str = None, **kwargs) -> List[Dict[str, Any]]:
        """List files and directories - must be implemented by subclasses"""
        pass
    
    


    # ____________________ Extra Mapping ____________________ #
    def create_file(self, file_path: str, content: Any, **kwargs) -> Dict[str, Any]:
        return self.save(file_path, content, **kwargs)
    
    def read_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return self.read(file_path, **kwargs)
    
    def list_files(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> Dict[str, Any]:
        return self.list(path, max_depth, include_hidden)
    
    def delete_file(self, file_path: str, **kwargs) -> Dict[str, Any]:
        return self.delete(file_path, **kwargs)
    
    def move_file(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return self.move(source, destination, **kwargs)
    
    def copy_file(self, source: str, destination: str, **kwargs) -> Dict[str, Any]:
        return self.copy(source, destination, **kwargs)


class LocalStorageHandler(FileStorageHandler):
    """
    Local filesystem storage implementation.
    Provides all file operations for local storage with default working directory.
    """
    
    def __init__(self, base_path: str = ".", **kwargs):
        """
        Initialize local storage handler.
        
        Args:
            base_path (str): Base directory for storage operations (default: current directory)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        super().__init__(base_path=base_path, **kwargs)
   
    def _initialize_storage(self):
        """Initialize local storage - ensure base directory exists"""
        try:
            # Ensure the base directory exists
            Path(self.base_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Local storage initialized with base path: {self.base_path}")
        except Exception as e:
            logger.error(f"Error initializing local storage: {str(e)}")
            raise
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from local filesystem"""
        try:
            with open(path, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to local filesystem"""
        try:
            # Ensure directory exists
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete file or directory from local filesystem"""
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink()
            elif path_obj.is_dir():
                shutil.rmtree(path_obj)
            else:
                return False
            return True
        except Exception as e:
            logger.error(f"Error deleting {path}: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files and directories in local filesystem"""
        try:
            if path is None:
                path = str(self.base_path)
            
            path_obj = Path(path)
            if not path_obj.exists() or not path_obj.is_dir():
                return []
            
            items = []
            
            def scan_directory(current_path: Path, current_depth: int):
                if current_depth > max_depth:
                    return
                
                try:
                    for item in current_path.iterdir():
                        # Skip hidden files if not included
                        if not include_hidden and item.name.startswith('.'):
                            continue
                        
                        try:
                            stat = item.stat()
                            item_info = {
                                "name": item.name,
                                "path": str(item),
                                "type": "directory" if item.is_dir() else "file",
                                "size_bytes": stat.st_size if item.is_file() else 0,
                                "size_mb": round(stat.st_size / (1024 * 1024), 2) if item.is_file() else 0,
                                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "extension": item.suffix.lower() if item.is_file() else "",
                                "is_hidden": item.name.startswith('.')
                            }
                            
                            items.append(item_info)
                            
                            # Recursively scan subdirectories
                            if item.is_dir() and current_depth < max_depth:
                                scan_directory(item, current_depth + 1)
                                
                        except (PermissionError, OSError):
                            # Skip files we can't access
                            continue
                            
                except (PermissionError, OSError) as e:
                    logger.warning(f"Error scanning directory {current_path}: {str(e)}")
            
            scan_directory(path_obj, 0)
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            return []
    
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in local filesystem"""
        return Path(path).exists()
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in local filesystem"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {path}: {str(e)}")
            return False
    


class SupabaseStorageHandler(FileStorageHandler):
    """
    Supabase remote storage implementation.
    Provides file operations via Supabase Storage API with environment-based configuration.
    """
    
    def __init__(self, bucket_name: str = None, base_path: str = "/", **kwargs):
        """
        Initialize Supabase storage handler.
        
        Args:
            bucket_name: Supabase storage bucket name (default: from environment or "default")
            base_path: Base path for storage operations (default: "/")
            **kwargs: Additional keyword arguments for parent class initialization
        """
        # Call parent constructor first
        super().__init__(base_path=base_path, **kwargs)
        
        # Get bucket name from environment or use default
        self.bucket_name = bucket_name or os.getenv("SUPABASE_BUCKET_STORAGE") or "default"
        self.supabase_url = os.getenv("SUPABASE_URL_STORAGE")
        self.supabase_key = os.getenv("SUPABASE_KEY_STORAGE")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Supabase configuration not found in environment variables. "
                "Please set SUPABASE_URL/SUPABASE_KEY environment variables."
            )
        
        # Initialize Supabase client
        try:
            from supabase import create_client, Client
            logger.info(f"Creating Supabase client with URL: {self.supabase_url[:30]}...")
            self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            logger.info(f"Successfully initialized Supabase client for bucket: {bucket_name}")
        except ImportError:
            raise ImportError(
                "Supabase Python client not installed. "
                "Please install it with: pip install supabase"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {str(e)}")
            raise Exception(f"Failed to initialize Supabase client: {str(e)}")
        
        # Initialize storage after all attributes are set
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize remote storage - verify bucket exists and is accessible"""
        # Check if required attributes are available
        if not hasattr(self, 'bucket_name') or not hasattr(self, 'supabase'):
            # If attributes aren't set yet, skip initialization
            # This will be called again after attributes are set in __init__
            return
        
        try:
            # Test bucket access by listing files (empty list is fine)
            logger.info(f"Testing bucket access for: {self.bucket_name}")
            self.supabase.storage.from_(self.bucket_name).list()
            logger.info(f"Successfully connected to Supabase bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify bucket access: {str(e)}")
            # Don't raise error as bucket might be empty or have different permissions
    
    def translate_in(self, file_path: str) -> str:
        """Resolve file path for remote storage"""
        # Use the translate_in method to combine base_path with file_path
        # For Supabase, we need to handle the special case where base_path is "/"
        if self.base_path == "/":
            # If base_path is "/", just clean the file_path
            return file_path.lstrip('/')
        else:
            # Use the standard translate_in method
            return super().translate_in(file_path)
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Download file from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).download(file_path)
            
            if isinstance(response, bytes):
                return response
            else:
                # If response is not bytes, try to convert
                return bytes(response) if response else b""
                
        except Exception as e:
            logger.error(f"Error reading file {path} from Supabase: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to Supabase Storage with smart insert/update logic"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Check if file already exists
            file_exists = self._exists_raw(file_path)
            
            if file_exists:
                # File exists, use update method
                logger.info(f"File {file_path} exists, using update method")
                response = self.supabase.storage.from_(self.bucket_name).update(
                    path=file_path,
                    file=content,
                    file_options={
                        "content-type": kwargs.get("content_type", "application/octet-stream"),
                        "upsert": "true"  # Ensure update works even if there are issues
                    }
                )
            else:
                # File doesn't exist, use upload method
                logger.info(f"File {file_path} doesn't exist, using upload method")
                response = self.supabase.storage.from_(self.bucket_name).upload(
                    path=file_path,
                    file=content,
                    file_options={"content-type": kwargs.get("content_type", "application/octet-stream")}
                )
            
            # Check if operation was successful
            if response and (not isinstance(response, dict) or response.get("error") is None):
                operation = "updated" if file_exists else "uploaded"
                logger.info(f"Successfully {operation} file to Supabase: {file_path}")
                return True
            else:
                logger.error(f"Operation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error writing file {path} to Supabase: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete file from Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Delete file from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).remove([file_path])
            
            # Check if deletion was successful
            # Supabase remove() returns an empty list [] when successful
            if response is not None:
                if isinstance(response, list):
                    # Empty list means successful deletion
                    logger.info(f"Successfully deleted file from Supabase: {file_path}")
                    return True
                elif isinstance(response, dict) and response.get("error") is None:
                    # Some responses might be dict format
                    logger.info(f"Successfully deleted file from Supabase: {file_path}")
                    return True
                else:
                    logger.error(f"Deletion failed: {response}")
                    return False
            else:
                logger.error(f"Deletion failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting {path} from Supabase: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files in Supabase Storage"""
        try:
            # Remove leading slash if present
            list_path = (path or self.base_path).lstrip('/')
            
            # List files from Supabase Storage
            response = self.supabase.storage.from_(self.bucket_name).list(list_path)
            
            items = []
            if response and isinstance(response, list):
                for item in response:
                    # Skip hidden files if not included
                    if not include_hidden and item.get('name', '').startswith('.'):
                        continue
                    
                    # Calculate full path
                    full_path = f"{list_path}/{item['name']}" if list_path else item['name']
                    
                    items.append({
                        "name": item.get('name', ''),
                        "path": full_path,
                        "type": "directory" if item.get('metadata', {}).get('mimetype') == 'application/x-directory' else "file",
                        "size_bytes": item.get('metadata', {}).get('size', 0),
                        "size_mb": round(item.get('metadata', {}).get('size', 0) / (1024 * 1024), 2),
                        "modified_time": item.get('updated_at', ''),
                        "extension": Path(item.get('name', '')).suffix.lower(),
                        "is_hidden": item.get('name', '').startswith('.'),
                        "mime_type": item.get('metadata', {}).get('mimetype', '')
                    })
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path} from Supabase: {str(e)}")
            return []
    
    def _exists_raw(self, path: str) -> bool:
        """Check if path exists in Supabase Storage"""
        try:
            # Remove leading slash if present
            file_path = path.lstrip('/')
            
            # Get the parent directory and filename
            parent_dir = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            
            # If no parent directory, check root
            if not parent_dir:
                parent_dir = ""
            
            try:
                # List files in the parent directory
                response = self.supabase.storage.from_(self.bucket_name).list(parent_dir)
                
                if response and isinstance(response, list):
                    # Check if our filename exists in the directory
                    for item in response:
                        if item.get('name') == file_name:
                            return True
                
                return False
                
            except Exception as e:
                logger.warning(f"Error listing directory {parent_dir}: {str(e)}")
                return False
                
        except Exception as e:
            logger.warning(f"Error checking if file {path} exists: {str(e)}")
            return False
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in Supabase Storage"""
        try:
            # Remove leading slash if present
            dir_path = path.lstrip('/')
            
            # Create a placeholder file to establish the directory
            placeholder_content = b"# Directory placeholder"
            placeholder_path = f"{dir_path}/.placeholder"
            
            response = self.supabase.storage.from_(self.bucket_name).upload(
                path=placeholder_path,
                file=placeholder_content,
                file_options={"content-type": "text/plain"}
            )
            
            # Check if upload was successful
            if response and not isinstance(response, dict) or response.get("error") is None:
                return True
            else:
                logger.error(f"Directory creation failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating directory {path} in Supabase: {str(e)}")
            return False
    
    

class NeonStorageHandler(FileStorageHandler):
    """
    Neon remote storage implementation.
    Provides file operations via Neon's S3-compatible storage API with environment-based configuration.
    """
    
    def __init__(self, bucket_name: str = None, base_path: str = "/",
                 endpoint_url: str = None, access_key_id: str = None,
                 secret_access_key: str = None, region: str = None, **kwargs):
        """
        Initialize Neon storage handler.
        
        Args:
            bucket_name: Neon storage bucket name (default: from NEON_BUCKET_STORAGE)
            base_path: Base path (key prefix) for storage operations (default: "/")
            endpoint_url: Branch storage endpoint (default: from AWS_ENDPOINT_URL_S3)
            access_key_id: Neon access key ID (default: from AWS_ACCESS_KEY_ID)
            secret_access_key: Neon secret access key (default: from AWS_SECRET_ACCESS_KEY)
            region: Region used for request signing (default: from AWS_REGION, then the endpoint host)
            **kwargs: Additional keyword arguments for parent class initialization
        """
        # Call parent constructor first
        super().__init__(base_path=base_path, **kwargs)
        
        self.bucket_name = bucket_name or os.getenv("NEON_BUCKET_STORAGE")
        self.endpoint_url = endpoint_url or os.getenv("AWS_ENDPOINT_URL_S3")
        access_key_id = access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = region or os.getenv("AWS_REGION") or self._region_from_endpoint(self.endpoint_url)
        
        if not self.bucket_name:
            raise ValueError(
                "Neon bucket name not found. "
                "Pass bucket_name or set the NEON_BUCKET_STORAGE environment variable."
            )
        if not self.endpoint_url or not access_key_id or not secret_access_key:
            raise ValueError(
                "Neon storage configuration not found in environment variables. "
                "Please set AWS_ENDPOINT_URL_S3/AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY environment variables."
            )
        
        # Initialize S3 client against the Neon branch endpoint
        try:
            import boto3
            from botocore.config import Config
            # Neon serves buckets under the branch endpoint, so requests must be path-style
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name=self.region,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
            logger.info(f"Successfully initialized Neon storage client for bucket: {self.bucket_name}")
        except ImportError:
            raise ImportError(
                "boto3 not installed. "
                "Please install it with: pip install boto3"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Neon storage client: {str(e)}")
            raise Exception(f"Failed to initialize Neon storage client: {str(e)}")
        
        # Initialize storage after all attributes are set
        self._initialize_storage()
    
    @staticmethod
    def _region_from_endpoint(endpoint_url: str) -> str:
        """Read the region from a host like br-x.storage.c-1.us-east-2.aws.neon.tech"""
        if endpoint_url:
            parts = endpoint_url.split("://")[-1].split("/")[0].split(".")
            if "storage" in parts and len(parts) > parts.index("storage") + 2:
                return parts[parts.index("storage") + 2]
        return "us-east-1"
    
    def _initialize_storage(self):
        """Initialize remote storage - verify bucket exists and is accessible"""
        # Check if required attributes are available
        if not hasattr(self, 'bucket_name') or not hasattr(self, 's3_client'):
            # If attributes aren't set yet, skip initialization
            # This will be called again after attributes are set in __init__
            return
        
        try:
            logger.info(f"Testing bucket access for: {self.bucket_name}")
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(f"Successfully connected to Neon bucket: {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Could not verify bucket access: {str(e)}")
            # Don't raise error as the credential might only allow object access
    
    def translate_in(self, file_path: str) -> str:
        """Resolve file path for remote storage"""
        if self.base_path == "/":
            # If base_path is "/", just clean the file_path
            return file_path.lstrip('/')
        else:
            return super().translate_in(file_path)
    
    @staticmethod
    def _is_not_found(error: Exception) -> bool:
        """Check whether a botocore error is a 404"""
        response = getattr(error, "response", None) or {}
        code = str(response.get("Error", {}).get("Code", ""))
        return code in ("404", "NoSuchKey", "NotFound")
    
    def _list_keys(self, prefix: str) -> List[Dict[str, Any]]:
        """List every object under a key prefix"""
        objects = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            objects.extend(page.get("Contents", []))
        return objects
    
    def _read_raw(self, path: str, **kwargs) -> bytes:
        """Read raw file content from Neon Object Storage"""
        try:
            key = path.lstrip('/')
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=key)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Error reading file {path} from Neon: {str(e)}")
            raise
    
    def _write_raw(self, path: str, content: bytes, **kwargs) -> bool:
        """Write raw file content to Neon Object Storage (PUT overwrites existing objects)"""
        try:
            key = path.lstrip('/')
            content_type = (
                kwargs.get("content_type")
                or mimetypes.guess_type(key)[0]
                or "application/octet-stream"
            )
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=content,
                ContentType=content_type,
            )
            logger.info(f"Successfully uploaded file to Neon: {key}")
            return True
        except Exception as e:
            logger.error(f"Error writing file {path} to Neon: {str(e)}")
            return False
    
    def _delete_raw(self, path: str) -> bool:
        """Delete a file, or every object under a directory prefix, from Neon Object Storage"""
        try:
            key = path.lstrip('/')
            if key and self._object_exists(key):
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
                logger.info(f"Successfully deleted file from Neon: {key}")
                return True
            
            # Treat the path as a directory and remove everything under it
            prefix = f"{key.rstrip('/')}/" if key else ""
            if not prefix:
                logger.error("Refusing to delete the whole bucket")
                return False
            objects = self._list_keys(prefix)
            if not objects:
                return False
            for i in range(0, len(objects), 1000):
                batch = [{"Key": obj["Key"]} for obj in objects[i:i + 1000]]
                self.s3_client.delete_objects(Bucket=self.bucket_name, Delete={"Objects": batch, "Quiet": True})
            logger.info(f"Successfully deleted {len(objects)} objects under {prefix} from Neon")
            return True
        except Exception as e:
            logger.error(f"Error deleting {path} from Neon: {str(e)}")
            return False
    
    def _list_raw(self, path: str = None, max_depth: int = 3, include_hidden: bool = False) -> List[Dict[str, Any]]:
        """List files and directories in Neon Object Storage"""
        try:
            list_path = (path or self.base_path).strip('/')
            prefix = f"{list_path}/" if list_path else ""
            
            items = []
            seen_dirs = set()
            for obj in self._list_keys(prefix):
                parts = obj["Key"][len(prefix):].split('/')
                
                # Directories implied by the key, up to max_depth levels below the listed path
                for depth in range(min(len(parts) - 1, max_depth + 1)):
                    dir_parts = parts[:depth + 1]
                    if not include_hidden and any(p.startswith('.') for p in dir_parts):
                        break
                    dir_path = prefix + '/'.join(dir_parts)
                    if dir_path not in seen_dirs:
                        seen_dirs.add(dir_path)
                        items.append({
                            "name": dir_parts[-1],
                            "path": dir_path,
                            "type": "directory",
                            "size_bytes": 0,
                            "size_mb": 0,
                            "modified_time": "",
                            "extension": "",
                            "is_hidden": dir_parts[-1].startswith('.')
                        })
                
                # Directory markers ("dir/") end with an empty name
                name = parts[-1]
                if not name or len(parts) - 1 > max_depth:
                    continue
                if not include_hidden and any(p.startswith('.') for p in parts):
                    continue
                
                size = obj.get("Size", 0)
                modified = obj.get("LastModified")
                items.append({
                    "name": name,
                    "path": obj["Key"],
                    "type": "file",
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "modified_time": modified.isoformat() if modified else "",
                    "extension": Path(name).suffix.lower(),
                    "is_hidden": name.startswith('.')
                })
            
            return items
            
        except Exception as e:
            logger.error(f"Error listing directory {path} from Neon: {str(e)}")
            return []
    
    def _object_exists(self, key: str) -> bool:
        """Check if a single object exists"""
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception as e:
            if self._is_not_found(e):
                return False
            raise
    
    def _exists_raw(self, path: str) -> bool:
        """Check if a file or directory prefix exists in Neon Object Storage"""
        try:
            key = path.lstrip('/')
            if not key:
                return True
            if self._object_exists(key):
                return True
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=f"{key.rstrip('/')}/", MaxKeys=1
            )
            return response.get("KeyCount", 0) > 0
        except Exception as e:
            logger.warning(f"Error checking if file {path} exists: {str(e)}")
            return False
    
    def _create_directory_raw(self, path: str) -> bool:
        """Create directory in Neon Object Storage"""
        try:
            dir_path = path.strip('/')
            if not dir_path:
                return True
            
            # Object storage has no real directories, so write an empty "dir/" marker object
            self.s3_client.put_object(Bucket=self.bucket_name, Key=f"{dir_path}/", Body=b"")
            return True
                
        except Exception as e:
            logger.error(f"Error creating directory {path} in Neon: {str(e)}")
            return False
