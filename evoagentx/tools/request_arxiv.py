import xml.etree.ElementTree as ET
from typing import Dict, Any, Optional, List
import re

from .tool import Tool, Toolkit
from .storage_handler import FileStorageHandler
from .request_base import RequestBase


class ArxivBase(RequestBase):
    """
    Extended RequestBase class for arXiv API interactions.
    Provides specialized methods for working with arXiv's Atom XML API.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.base_url = "http://export.arxiv.org/api/query"
        self.atom_namespace = "http://www.w3.org/2005/Atom"
        self.arxiv_namespace = "http://arxiv.org/schemas/atom"
        self.opensearch_namespace = "http://a9.com/-/spec/opensearch/1.1/"
    
    def search_arxiv(self, search_query: str = None, id_list: List[str] = None, 
                     start: int = 0, max_results: int = 10) -> Dict[str, Any]:
        """
        Search arXiv using the API and return structured results.
        
        Args:
            search_query: Search query string (e.g., "all:electron", "cat:cs.AI")
            id_list: List of arXiv IDs to retrieve
            start: Starting index for results
            max_results: Maximum number of results to return
            
        Returns:
            Dictionary containing parsed search results
        """
        # Build query parameters
        params = {
            'start': start,
            'max_results': max_results
        }
        
        if search_query:
            params['search_query'] = search_query
        
        if id_list:
            params['id_list'] = ','.join(id_list)
        
        try:
            # Make the HTTP request
            response = self.request(
                url=self.base_url,
                method='GET',
                params=params
            )
            
            # Parse the XML response
            return self._parse_atom_response(response.text)
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'query': search_query or str(id_list)
            }
    
    def _parse_atom_response(self, xml_content: str) -> Dict[str, Any]:
        """
        Parse the Atom XML response from arXiv API.
        
        Args:
            xml_content: Raw XML content from the API response
            
        Returns:
            Dictionary with parsed paper information
        """
        try:
            # Parse XML
            root = ET.fromstring(xml_content)
            
            # Register namespaces
            namespaces = {
                'atom': self.atom_namespace,
                'arxiv': self.arxiv_namespace,
                'opensearch': self.opensearch_namespace
            }
            
            # Extract metadata
            total_results = root.find('.//opensearch:totalResults', namespaces)
            start_index = root.find('.//opensearch:startIndex', namespaces)
            items_per_page = root.find('.//opensearch:itemsPerPage', namespaces)
            
            result = {
                'success': True,
                'total_results': int(total_results.text) if total_results is not None else 0,
                'start_index': int(start_index.text) if start_index is not None else 0,
                'items_per_page': int(items_per_page.text) if items_per_page is not None else 0,
                'papers': []
            }
            
            # Extract paper entries
            entries = root.findall('.//atom:entry', namespaces)
            
            for entry in entries:
                paper = self._parse_paper_entry(entry, namespaces)
                result['papers'].append(paper)
            
            return result
            
        except ET.ParseError as e:
            return {
                'success': False,
                'error': f'XML parsing error: {str(e)}',
                'raw_content': xml_content[:500] + '...' if len(xml_content) > 500 else xml_content
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _parse_paper_entry(self, entry, namespaces) -> Dict[str, Any]:
        """
        Parse a single paper entry from the XML.
        
        Args:
            entry: XML element for a paper entry
            namespaces: Namespace mappings
            
        Returns:
            Dictionary with paper information
        """
        paper = {}
        
        # Basic information
        paper['id'] = self._get_text(entry, 'atom:id', namespaces)
        paper['title'] = self._get_text(entry, 'atom:title', namespaces, clean=True)
        paper['summary'] = self._get_text(entry, 'atom:summary', namespaces, clean=True)
        paper['published'] = self._get_text(entry, 'atom:published', namespaces)
        paper['updated'] = self._get_text(entry, 'atom:updated', namespaces)
        
        # Extract arXiv ID from the full ID URL
        if paper['id']:
            paper['arxiv_id'] = paper['id'].split('/')[-1]
        
        # Authors
        authors = entry.findall('.//atom:author', namespaces)
        paper['authors'] = []
        for author in authors:
            name = self._get_text(author, 'atom:name', namespaces)
            if name:
                paper['authors'].append(name)
        
        # Categories
        categories = entry.findall('.//atom:category', namespaces)
        paper['categories'] = []
        for category in categories:
            term = category.get('term')
            if term:
                paper['categories'].append(term)
        
        # Primary category
        primary_cat = entry.find('.//arxiv:primary_category', namespaces)
        if primary_cat is not None:
            paper['primary_category'] = primary_cat.get('term')
        
        # Links (PDF, HTML)
        links = entry.findall('.//atom:link', namespaces)
        paper['links'] = {}
        for link in links:
            rel = link.get('rel')
            href = link.get('href')
            title = link.get('title')
            
            if rel == 'alternate':
                paper['links']['html'] = href
            elif title == 'pdf':
                paper['links']['pdf'] = href
        
        # arXiv-specific fields
        paper['comment'] = self._get_text(entry, 'arxiv:comment', namespaces)
        paper['journal_ref'] = self._get_text(entry, 'arxiv:journal_ref', namespaces)
        paper['doi'] = self._get_text(entry, 'arxiv:doi', namespaces)
        
        # Map field names for better API
        # Use the HTML link as the main URL, fallback to constructing from arxiv_id
        if paper.get('links', {}).get('html'):
            paper['url'] = paper['links']['html']
        elif paper.get('arxiv_id'):
            paper['url'] = f"https://arxiv.org/abs/{paper['arxiv_id']}"
        else:
            paper['url'] = ''
        
        paper['published_date'] = paper.pop('published', '')
        paper['updated_date'] = paper.pop('updated', '')
        
        # Remove the old id field since we're replacing it with url
        paper.pop('id', None)
        
        return paper
    
    def _get_text(self, element, xpath, namespaces, clean=False) -> str:
        """
        Helper method to extract text from XML elements.
        
        Args:
            element: XML element to search in
            xpath: XPath expression
            namespaces: Namespace mappings
            clean: Whether to clean whitespace
            
        Returns:
            Text content or empty string
        """
        found = element.find(xpath, namespaces)
        if found is not None:
            text = found.text or ''
            if clean:
                # Clean up whitespace and newlines
                text = re.sub(r'\s+', ' ', text.strip())
            return text
        return ''
    
    def download_pdf(self, pdf_url: str, save_path: str, storage_handler: FileStorageHandler = None) -> Dict[str, Any]:
        """
        Download a PDF from arXiv.
        
        Args:
            pdf_url: URL of the PDF to download
            save_path: Local path to save the PDF
            storage_handler: Storage handler for file operations
            
        Returns:
            Dictionary with download status
        """
        try:
            response = self.request(url=pdf_url, method='GET')
            
            # Get the PDF content
            pdf_content = response.content
            
            # Save the PDF content using storage handler
            result = storage_handler.save(save_path, pdf_content)
            if result["success"]:
                return {
                    'success': True,
                    'file_path': save_path,
                    'size': len(pdf_content),
                    'url': pdf_url,
                    'storage_handler': type(storage_handler).__name__
                }
            else:
                return {
                    'success': False,
                    'error': f"Failed to save PDF: {result.get('error', 'Unknown error')}",
                    'url': pdf_url,
                    'save_path': save_path
                }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': pdf_url
            }


class ArxivSearchTool(Tool):
    """Tool for searching papers on arXiv."""
    
    name: str = "arxiv_search"
    description: str = "Search for academic papers on arXiv using queries or paper IDs"
    inputs: Dict[str, Dict[str, str]] = {
        "search_query": {
            "type": "string",
            "description": "Search query (e.g., 'all:machine learning', 'cat:cs.AI', 'au:smith')"
        },
        "id_list": {
            "type": "array",
            "description": "List of arXiv IDs to retrieve (e.g., ['1706.03762', '1810.04805'])",
            "items": {"type": "string"}
        },
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results to return (default: 10)"
        },
        "start": {
            "type": "integer",
            "description": "Starting index for pagination (default: 0)"
        }
    }
    required: Optional[List[str]] = []
    
    def __init__(self, arxiv_base: ArxivBase = None):
        super().__init__()
        self.arxiv_base = arxiv_base
    
    def __call__(self, search_query: str = None, id_list: list = None, 
                 max_results: int = 10, start: int = 0) -> Dict[str, Any]:
        """
        Search arXiv for papers.
        
        Args:
            search_query: Search query string
            id_list: List of arXiv IDs
            max_results: Maximum results to return
            start: Starting index for pagination
            
        Returns:
            Dictionary with search results
        """
        if not search_query and not id_list:
            return {
                'success': False,
                'error': 'Either search_query or id_list must be provided'
            }
        
        return self.arxiv_base.search_arxiv(
            search_query=search_query,
            id_list=id_list,
            start=start,
            max_results=max_results
        )


class ArxivDownloadTool(Tool):
    """Tool for downloading papers from arXiv."""
    
    name: str = "arxiv_download"
    description: str = "Download PDF papers from arXiv"
    inputs: Dict[str, Dict[str, str]] = {
        "pdf_url": {
            "type": "string",
            "description": "URL of the PDF to download"
        },
        "save_path": {
            "type": "string",
            "description": "Local path to save the PDF file"
        }
    }
    required: Optional[List[str]] = ["pdf_url", "save_path"]
    
    def __init__(self, arxiv_base: ArxivBase = None, storage_handler: FileStorageHandler = None):
        super().__init__()
        self.arxiv_base = arxiv_base
        self.storage_handler = storage_handler

    def __call__(self, pdf_url: str, save_path: str) -> Dict[str, Any]:
        """
        Download a PDF from arXiv.
        
        Args:
            pdf_url: URL of the PDF
            save_path: Where to save the file
            
        Returns:
            Dictionary with download status
        """
        return self.arxiv_base.download_pdf(pdf_url, save_path, self.storage_handler)


class ArxivToolkit(Toolkit):
    def __init__(self, name: str = "ArxivToolkit", storage_handler: FileStorageHandler = None):
        # Initialize storage handler if not provided
        if storage_handler is None:
            from .storage_handler import LocalStorageHandler
            storage_handler = LocalStorageHandler()
        
        # Create the shared arxiv base instance
        arxiv_base = ArxivBase()
        
        # Create tools with the shared base and storage handler
        tools = [
            ArxivSearchTool(arxiv_base=arxiv_base),
            ArxivDownloadTool(arxiv_base=arxiv_base, storage_handler=storage_handler)
        ]
        
        # Initialize parent with tools
        super().__init__(name=name, tools=tools)
        
        # Store arxiv_base as instance variable
        self.arxiv_base = arxiv_base
        self.storage_handler = storage_handler 