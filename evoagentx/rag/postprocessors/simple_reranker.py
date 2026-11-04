from typing import List, Optional

from llama_index.core.schema import NodeWithScore
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

from .base import BasePostprocessor
from evoagentx.rag.schema import Query, RagResult, Corpus, Chunk


class SimpleReranker(BasePostprocessor):
    """Post-processor for reranking retrieval results."""
    
    def __init__(
        self,
        similarity_cutoff: Optional[float] = None,
        keyword_filters: Optional[List[str]] = None
    ):
        super().__init__()
        self.postprocessors = []
        if similarity_cutoff:
            self.postprocessors.append(SimilarityPostprocessor(similarity_cutoff=similarity_cutoff))
        if keyword_filters:
            self.postprocessors.append(KeywordNodePostprocessor(required_keywords=keyword_filters))
    
    def postprocess(self, query: Query, results: List[RagResult]) -> RagResult:
        try:
            # If no postprocessors, just combine results
            if not self.postprocessors:
                corpus = Corpus()
                scores = []
                for result in results:
                    for chunk in result.corpus.chunks:
                        corpus.add_chunk(chunk)
                    scores.extend(result.scores)
                
                final_result = RagResult(
                    corpus=corpus,
                    scores=scores,
                    metadata={"query": query.query_str, "postprocessor": "simple_passthrough"}
                )
                self.logger.info(f"Simple passthrough: {len(corpus.chunks)} chunks")
                return final_result
            
            # Create a mapping from chunk to original chunk for later reconstruction
            chunk_to_original = {}
            nodes = []
            
            for result in results:
                for chunk, score in zip(result.corpus.chunks, result.scores):
                    node = chunk.to_llama_node()
                    # IDs are local to a corpus. Use a request-local identity
                    # while postprocessors combine nodes from several corpora.
                    node.id_ = str(len(nodes))
                    nodes.append(NodeWithScore(node=node, score=score))
                    chunk_to_original[node.id_] = chunk
            
            # Apply postprocessors
            for postprocessor in self.postprocessors:
                nodes = postprocessor.postprocess_nodes(nodes)

            # Reconstruct corpus with original chunk types
            corpus = Corpus()
            scores = []
            for score_node in nodes:
                original_chunk = chunk_to_original.get(score_node.node.id_)
                
                if original_chunk:
                    # Use the original chunk and update its similarity score
                    original_chunk.metadata.similarity_score = score_node.score or 0.0
                    corpus.add_chunk(original_chunk)
                    scores.append(score_node.score or 0.0)
                else:
                    # Fallback: try to determine chunk type from first result
                    chunk_class = type(results[0].corpus.chunks[0]) if results and results[0].corpus.chunks else Chunk
                    try:
                        chunk = chunk_class.from_llama_node(score_node.node)
                        chunk.metadata.similarity_score = score_node.score or 0.0
                        corpus.add_chunk(chunk)
                        scores.append(score_node.score or 0.0)
                    except Exception as e:
                        self.logger.warning(f"Failed to reconstruct chunk from node: {e}")
                        continue
            
            result = RagResult(
                corpus=corpus,
                scores=scores,
                metadata={"query": query.query_str, "postprocessor": "reranker"}
            )
            self.logger.info(f"Reranked to {len(corpus.chunks)} chunks")
            return result
        except Exception as e:
            self.logger.error(f"Reranking failed: {str(e)}")
            raise