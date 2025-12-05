"""
Vector Store implementation using ChromaDB for shared agent memory.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid
from config.settings import get_settings


class VectorStore:
    """
    Vector database for storing and retrieving agent findings.
    Uses ChromaDB for efficient similarity search.
    """

    def __init__(self, collection_name: str = "financial_research"):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection
        """
        self.settings = get_settings()
        self.collection_name = collection_name

        # Initialize ChromaDB client with new API
        self.client = chromadb.PersistentClient(
            path=self.settings.vector_db_path
        )

        # Use sentence-transformers embedding function to avoid ONNX issues
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Check if collection exists to avoid embedding function conflicts
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(
                name=self.collection_name
            )
        except Exception:
            # Collection doesn't exist, create it with embedding function
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Financial research agent findings"},
                embedding_function=self.embedding_fn
            )

    def add_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> str:
        """
        Add a document to the vector store.

        Args:
            content: Text content to store
            metadata: Metadata about the document (agent, ticker, timestamp, etc.)
            document_id: Optional custom document ID

        Returns:
            Document ID
        """
        if document_id is None:
            document_id = str(uuid.uuid4())

        # Add timestamp if not present
        if 'timestamp' not in metadata:
            metadata['timestamp'] = datetime.now().isoformat()

        # Add to collection
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[document_id]
        )

        return document_id

    def add_batch(
        self,
        contents: List[str],
        metadatas: List[Dict[str, Any]],
        document_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add multiple documents to the vector store in batch.

        Args:
            contents: List of text contents
            metadatas: List of metadata dictionaries
            document_ids: Optional list of custom document IDs

        Returns:
            List of document IDs
        """
        if document_ids is None:
            document_ids = [str(uuid.uuid4()) for _ in contents]

        # Add timestamps
        for metadata in metadatas:
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.now().isoformat()

        # Add to collection
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            ids=document_ids
        )

        return document_ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents.

        Args:
            query_text: Text to search for
            n_results: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            Dictionary containing documents, metadatas, and distances
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=filter_metadata
        )

        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else [],
            'distances': results['distances'][0] if results['distances'] else []
        }

    def get_by_ticker(self, ticker: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Retrieve all documents for a specific ticker.

        Args:
            ticker: Stock ticker symbol
            n_results: Maximum number of results

        Returns:
            Dictionary containing documents and metadatas
        """
        results = self.collection.query(
            query_texts=[ticker],
            n_results=n_results,
            where={"ticker": ticker}
        )

        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else []
        }

    def get_by_agent(self, agent_name: str, n_results: int = 10) -> Dict[str, Any]:
        """
        Retrieve documents created by a specific agent.

        Args:
            agent_name: Name of the agent
            n_results: Maximum number of results

        Returns:
            Dictionary containing documents and metadatas
        """
        results = self.collection.query(
            query_texts=[agent_name],
            n_results=n_results,
            where={"agent": agent_name}
        )

        return {
            'documents': results['documents'][0] if results['documents'] else [],
            'metadatas': results['metadatas'][0] if results['metadatas'] else []
        }

    def clear_ticker(self, ticker: str) -> None:
        """
        Clear all documents for a specific ticker.

        Args:
            ticker: Stock ticker symbol
        """
        # Get all documents for ticker
        results = self.collection.get(
            where={"ticker": ticker}
        )

        if results['ids']:
            self.collection.delete(ids=results['ids'])

    def clear_all(self) -> None:
        """Clear all documents from the vector store."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "Financial research agent findings"},
            embedding_function=self.embedding_fn
        )

    def get_context(self, ticker: str, agent: Optional[str] = None) -> str:
        """
        Get formatted context for a ticker, optionally filtered by agent.

        Args:
            ticker: Stock ticker symbol
            agent: Optional agent name to filter by

        Returns:
            Formatted context string
        """
        filter_dict = {"ticker": ticker}
        if agent:
            filter_dict["agent"] = agent

        results = self.collection.query(
            query_texts=[ticker],
            n_results=20,
            where=filter_dict
        )

        if not results['documents'] or not results['documents'][0]:
            return f"No context found for {ticker}"

        context_parts = []
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            agent_name = metadata.get('agent', 'Unknown')
            timestamp = metadata.get('timestamp', 'Unknown')
            context_parts.append(f"[{agent_name} - {timestamp}]\n{doc}\n")

        return "\n---\n".join(context_parts)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary containing collection statistics
        """
        count = self.collection.count()

        # Get unique tickers and agents
        all_data = self.collection.get()
        metadatas = all_data.get('metadatas', [])

        tickers = set()
        agents = set()
        for metadata in metadatas:
            if 'ticker' in metadata:
                tickers.add(metadata['ticker'])
            if 'agent' in metadata:
                agents.add(metadata['agent'])

        return {
            'total_documents': count,
            'unique_tickers': len(tickers),
            'unique_agents': len(agents),
            'tickers': list(tickers),
            'agents': list(agents)
        }
