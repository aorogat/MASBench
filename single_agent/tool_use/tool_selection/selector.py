"""
Tool Selector

Centralized LLM-based tool selection that runs before framework binding.
Caches selections per query for fairness and reproducibility.
"""
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool


class ToolSelector:
    """
    LLM-based tool selector that selects top-k most relevant tools for a query.
    
    Features:
    - Uses LLM to select tools based on query relevance
    - Caches selections per query for reproducibility
    - Ensures all frameworks use the same tool set for the same query
    - Respects LLM tool limits (e.g., OpenAI's 128-tool constraint)
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tools: int = 120,
        cache_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize ToolSelector.
        
        Args:
            model: OpenAI model name for tool selection (default: gpt-4o-mini)
            temperature: Temperature for LLM (default: 0.0 for reproducibility)
            max_tools: Maximum number of tools to select (default: 120, under OpenAI's 128 limit)
            cache_dir: Directory to cache tool selections (default: ./tool_selection_cache)
            verbose: Whether to print debug information
        """
        self.model = model
        self.temperature = temperature
        self.max_tools = max_tools
        self.verbose = verbose
        
        # Initialize LLM for tool selection
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Set up cache directory
        if cache_dir is None:
            # Default to tool_selection_cache in the same directory as this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(current_dir, "tool_selection_cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        if self.verbose:
            print(f"[ToolSelector] Initialized with model={model}, max_tools={max_tools}")
            print(f"[ToolSelector] Cache directory: {self.cache_dir}")
    
    def _get_query_hash(self, query: str) -> str:
        """
        Generate a hash for the query + max_tools + model to use as cache key.
        
        This ensures cache is k-aware: different max_tools values produce different cache keys.
        
        Args:
            query: The query string
            
        Returns:
            SHA256 hash of (query, max_tools, model)
        """
        # Include max_tools and model in hash to make cache k-aware
        cache_key = f"{query}|max_tools={self.max_tools}|model={self.model}"
        return hashlib.sha256(cache_key.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, query: str) -> str:
        """
        Get the cache file path for a query.
        
        Cache key includes max_tools and model, so different k values produce different cache files.
        
        Args:
            query: The query string
            
        Returns:
            Path to cache file
        """
        query_hash = self._get_query_hash(query)
        return os.path.join(self.cache_dir, f"{query_hash}.json")
    
    def _load_from_cache(self, query: str) -> Optional[List[str]]:
        """
        Load tool selection from cache if it exists and matches current max_tools.
        
        Args:
            query: The query string
            
        Returns:
            List of selected tool names, or None if not cached or cache is invalid
        """
        cache_path = self._get_cache_path(query)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    # Verify query matches (safety check)
                    if cache_data.get('query') != query:
                        if self.verbose:
                            print(f"[ToolSelector] Cache query mismatch, treating as cache miss")
                        return None
                    
                    # CRITICAL: Verify max_tools matches (cache must be k-aware)
                    cached_max_tools = cache_data.get('max_tools')
                    if cached_max_tools != self.max_tools:
                        if self.verbose:
                            print(f"[ToolSelector] Cache max_tools mismatch (cached: {cached_max_tools}, current: {self.max_tools}), treating as cache miss")
                        return None
                    
                    # Verify model matches (for consistency)
                    cached_model = cache_data.get('model')
                    if cached_model != self.model:
                        if self.verbose:
                            print(f"[ToolSelector] Cache model mismatch (cached: {cached_model}, current: {self.model}), treating as cache miss")
                        return None
                    
                    selected_tools = cache_data.get('selected_tools', [])
                    # Don't use empty cache entries - treat as cache miss
                    if len(selected_tools) == 0:
                        if self.verbose:
                            print(f"[ToolSelector] Cache contains empty selection, treating as cache miss")
                        return None
                    
                    # CRITICAL: Enforce max_tools even if cache has more (defensive programming)
                    if len(selected_tools) > self.max_tools:
                        if self.verbose:
                            print(f"[ToolSelector] Cache has {len(selected_tools)} tools, truncating to max_tools={self.max_tools}")
                        selected_tools = selected_tools[:self.max_tools]
                    
                    if self.verbose:
                        print(f"[ToolSelector] Loaded {len(selected_tools)} tools from cache")
                    return selected_tools
            except Exception as e:
                if self.verbose:
                    print(f"[ToolSelector] Error loading cache: {e}")
        return None
    
    def _save_to_cache(self, query: str, selected_tools: List[str]) -> None:
        """
        Save tool selection to cache.
        
        Only saves non-empty selections to avoid caching failures.
        
        Args:
            query: The query string
            selected_tools: List of selected tool names
        """
        # Don't save empty selections to cache
        if len(selected_tools) == 0:
            if self.verbose:
                print(f"[ToolSelector] Skipping cache save: empty selection (likely fallback failed)")
            return
        
        cache_path = self._get_cache_path(query)
        cache_data = {
            'query': query,
            'selected_tools': selected_tools,
            'model': self.model,
            'max_tools': self.max_tools
        }
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            if self.verbose:
                print(f"[ToolSelector] Saved {len(selected_tools)} tools to cache")
        except Exception as e:
            if self.verbose:
                print(f"[ToolSelector] Error saving cache: {e}")
    
    def _create_tool_metadata(self, tools: List[BaseTool]) -> List[Dict[str, str]]:
        """
        Create metadata list for tools (name only for speed).
        
        Args:
            tools: List of LangChain tools
            
        Returns:
            List of tool metadata dictionaries (name only)
        """
        metadata = []
        for tool in tools:
            metadata.append({
                'name': tool.name
            })
        return metadata
    
    def _select_tools_with_llm(
        self,
        query: str,
        tool_metadata: List[Dict[str, str]]
    ) -> List[str]:
        """
        Use LLM to select the most relevant tools for a query.
        
        Uses only tool names (no descriptions) for speed.
        
        Args:
            query: The user query
            tool_metadata: List of tool metadata (name only)
            
        Returns:
            List of selected tool names (ordered by relevance)
        """
        # Create prompt for tool selection (names only for speed)
        tools_text = "\n".join([
            f"{i+1}. {tool['name']}"
            for i, tool in enumerate(tool_metadata)
        ])
        
        system_prompt = f"""You are a tool selection assistant. Given a user query and a list of available tool names, select the {self.max_tools} most relevant tools that could help answer the query.

Return ONLY a JSON array of tool names in order of relevance (most relevant first). Do not include any explanation or other text.

Example format:
["tool_name_1", "tool_name_2", "tool_name_3", ...]

Available tools (names only):
{tools_text}"""
        
        user_prompt = f"User query: {query}\n\nSelect the {self.max_tools} most relevant tools and return them as a JSON array of tool names."
        
        if self.verbose:
            print(f"[ToolSelector] Selecting tools with LLM (query: {query[:100]}...)")
            print(f"[ToolSelector] Total tools available: {len(tool_metadata)}")
        
        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        response = self.llm.invoke(messages)
        response_text = response.content.strip()
        
        # Parse JSON response
        try:
            # Extract JSON array from response (handle markdown code blocks)
            if response_text.startswith("```"):
                # Remove markdown code block markers
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            elif response_text.startswith("```json"):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Try to parse JSON - handle truncated responses
            try:
                selected_tools = json.loads(response_text)
            except json.JSONDecodeError as json_err:
                # Response might be truncated - try to extract valid JSON
                if self.verbose:
                    print(f"[ToolSelector] JSON parse error (possibly truncated): {json_err}")
                
                # Try to find the start of a JSON array
                start_idx = response_text.find('[')
                if start_idx == -1:
                    raise json_err
                
                # Try to extract up to the truncation point
                # Find the last complete tool name before the error
                truncated_text = response_text[:json_err.pos] if hasattr(json_err, 'pos') else response_text
                
                # Try to find a closing bracket
                last_bracket = truncated_text.rfind(']')
                if last_bracket > start_idx:
                    # Extract up to the last complete bracket
                    truncated_text = truncated_text[:last_bracket + 1]
                    try:
                        selected_tools = json.loads(truncated_text)
                    except:
                        raise json_err
                else:
                    # No closing bracket - try to extract tool names manually
                    # Look for patterns like "tool_name" or "tool_name",
                    import re
                    tool_name_pattern = r'"([^"]+)"'
                    matches = re.findall(tool_name_pattern, truncated_text)
                    if matches:
                        selected_tools = matches[:self.max_tools]
                    else:
                        raise json_err
            
            if not isinstance(selected_tools, list):
                raise ValueError("Response is not a list")
            
            # Validate tool names exist
            tool_names = {tool['name'] for tool in tool_metadata}
            valid_tools = [name for name in selected_tools if name in tool_names]
            
            # Limit to max_tools
            valid_tools = valid_tools[:self.max_tools]
            
            if self.verbose:
                print(f"[ToolSelector] Selected {len(valid_tools)} tools")
            
            return valid_tools
            
        except Exception as e:
            if self.verbose:
                print(f"[ToolSelector] Error parsing LLM response: {e}")
                print(f"[ToolSelector] Response length: {len(response_text)} chars")
                print(f"[ToolSelector] Response preview: {response_text[:200]}...")
            
            # Fallback: use keyword-based selection
            if self.verbose:
                print(f"[ToolSelector] Falling back to keyword-based selection...")
            return self._fallback_keyword_selection(query, tool_metadata)
    
    def select_tools(
        self,
        query: str,
        all_tools: List[BaseTool]
    ) -> List[BaseTool]:
        """
        Select the most relevant tools for a query.
        
        Uses cache if available, otherwise calls LLM and caches the result.
        
        Args:
            query: The user query
            all_tools: List of all available tools
            
        Returns:
            List of selected tools (ordered by relevance, max max_tools)
        """
        # Check cache first
        cached_tool_names = self._load_from_cache(query)
        
        if cached_tool_names is not None:
            # Load tools from cache
            # Cache may contain original names (before sanitization), so we need to match by:
            # 1. Direct name match (sanitized name)
            # 2. Original name from metadata
            tool_map_by_name = {tool.name: tool for tool in all_tools}
            tool_map_by_original = {}
            for tool in all_tools:
                if hasattr(tool, 'metadata') and isinstance(tool.metadata, dict):
                    original_name = tool.metadata.get('original_name')
                    if original_name:
                        tool_map_by_original[original_name] = tool
            
            selected_tools = []
            for cached_name in cached_tool_names:
                # Try direct name match first (sanitized)
                if cached_name in tool_map_by_name:
                    selected_tools.append(tool_map_by_name[cached_name])
                # Try original name match (from metadata)
                elif cached_name in tool_map_by_original:
                    selected_tools.append(tool_map_by_original[cached_name])
                # Name not found (tool might have been removed or renamed)
                elif self.verbose:
                    print(f"[ToolSelector] Warning: Cached tool name '{cached_name}' not found in current tools")
            
            # CRITICAL: Enforce max_tools limit even after loading from cache
            # This is a defensive check in case cache validation missed something
            if len(selected_tools) > self.max_tools:
                if self.verbose:
                    print(f"[ToolSelector] Truncating {len(selected_tools)} cached tools to max_tools={self.max_tools}")
                selected_tools = selected_tools[:self.max_tools]
            
            if self.verbose:
                print(f"[ToolSelector] Using cached selection: {len(selected_tools)} tools (from {len(cached_tool_names)} cached names)")
            return selected_tools
        
        # If not in cache, use LLM to select
        tool_metadata = self._create_tool_metadata(all_tools)
        selected_tool_names = self._select_tools_with_llm(query, tool_metadata)
        
        # Ensure we have tools (fallback should have provided some, but double-check)
        if len(selected_tool_names) == 0:
            if self.verbose:
                print(f"[ToolSelector] Warning: No tools selected from LLM, using keyword fallback")
            selected_tool_names = self._fallback_keyword_selection(query, tool_metadata)
        
        # Save to cache (only if we have tools - _save_to_cache will skip if empty)
        self._save_to_cache(query, selected_tool_names)
        
        # Return selected tools
        tool_map = {tool.name: tool for tool in all_tools}
        selected_tools = [
            tool_map[name] for name in selected_tool_names
            if name in tool_map
        ]
        
        # CRITICAL: Enforce max_tools limit (defensive check)
        if len(selected_tools) > self.max_tools:
            if self.verbose:
                print(f"[ToolSelector] Truncating {len(selected_tools)} selected tools to max_tools={self.max_tools}")
            selected_tools = selected_tools[:self.max_tools]
        
        return selected_tools
    
    def _fallback_keyword_selection(
        self,
        query: str,
        tool_metadata: List[Dict[str, str]]
    ) -> List[str]:
        """
        Fallback keyword-based tool selection when LLM parsing fails.
        
        Args:
            query: The user query
            tool_metadata: List of tool metadata (name only)
            
        Returns:
            List of selected tool names
        """
        query_lower = query.lower()
        query_words = set(word for word in query_lower.split() if len(word) > 2)
        
        # Score tools based on keyword matches in tool names
        tool_scores = []
        for tool in tool_metadata:
            tool_name_lower = tool['name'].lower()
            score = 0
            
            # Check for keyword matches
            for word in query_words:
                if word in tool_name_lower:
                    score += 1
            
            if score > 0:
                tool_scores.append((score, tool['name']))
        
        # Sort by score (descending) and take top max_tools
        tool_scores.sort(key=lambda x: x[0], reverse=True)
        selected_tools = [name for _, name in tool_scores[:self.max_tools]]
        
        if self.verbose:
            print(f"[ToolSelector] Keyword fallback selected {len(selected_tools)} tools")
        
        return selected_tools

