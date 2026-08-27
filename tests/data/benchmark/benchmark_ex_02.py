from evoagentx.config import Config
from evoagentx.models import OpenAILLMConfig, OpenAILLM
from evoagentx.benchmark.worfbench import WorfBench
from evoagentx.evaluators import Evaluator
from evoagentx.workflow import QAActionGraph 
from evoagentx.core.callbacks import suppress_logger_info
import os
from dotenv import load_dotenv
from evoagentx.core.logging import logger
from evoagentx.benchmark.benchmark import Benchmark

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
if not HF_TOKEN:
    logger.warning("HF_TOKEN not found, may fail to access restricted datasets")

# Configure OpenAI LLM
llm_config = OpenAILLMConfig(model="gpt-4", openai_key=OPENAI_API_KEY)
llm = OpenAILLM(config=llm_config)

# Initialize WorfBench benchmark
class MessageBenchmark(Benchmark):
    """
    Adapt dataset in messages format, automatically extract last user/assistant round.
    """
    def __init__(self, path: str, mode: str = "train"):
        super().__init__(name="MessageBenchmark", path=path, mode=mode)

    def _load_data(self):
        # Load only train data
        import json
        file_path = os.path.join(self.path, "worfbench_train.json")
        with open(file_path, "r", encoding="utf-8") as f:
            self._train_data = json.load(f)

    def _get_label(self, example):
        # Get last assistant message
        return [m["content"] for m in example["messages"] if m["role"] == "assistant"][-1]

    def _get_id(self, example):
        # Use source + last user message content as unique id
        user_msg = [m["content"] for m in example["messages"] if m["role"] == "user"][-1]
        return example.get("source", "") + "_" + user_msg[:20]

    def evaluate(self, prediction, label):
        from evoagentx.benchmark.measures import exact_match_score, f1_score, acc_score
        em = exact_match_score(prediction, label)
        f1 = f1_score(prediction, label)
        acc = acc_score(prediction, [label])
        return {"em": em, "f1": f1, "acc": acc}

benchmark = MessageBenchmark(path="./data/worfbench", mode="train")

# Define workflow graph
workflow = QAActionGraph(
    llm_config=llm_config,
    description="This workflow aims to address multi-hop QA tasks."
)

def collate_func(example: dict) -> dict:
    user_msg = [m["content"] for m in example["messages"] if m["role"] == "user"][-1]
    assistant_msgs = [m["content"] for m in example["messages"] if m["role"] == "assistant"]
    example_workflow = assistant_msgs[-2] if len(assistant_msgs) >= 2 else ""
    
    # Improved prompt, emphasizing graph structure importance
    prompt = (
        f"{user_msg}\n\n"
        "Please strictly output in the following format:\n"
        "<thought>Your reasoning process</thought>\n"
        "<answer>\n"
        "Node:\n1: ...\n2: ...\nEdge: (START,1) (1,2) ... (n,END)\n"
        "</answer>\n"
        "Important notes:\n"
        "1. Carefully analyze task dependencies, some steps can be executed in parallel\n"
        "2. Ensure edge connections correctly reflect task dependencies\n"
        "3. Node count should match task complexity\n"
        "4. Use (START,1) for start, (n,END) for end\n"
        "Example:\n"
        f"{example_workflow}\n"
        "Only output your workflow, no extra content."
    )
    return {"problem": prompt}

def output_postprocess_func(output: dict) -> str:
    if isinstance(output, dict) and "answer" in output:
        out = output["answer"]
    else:
        out = str(output)
    
    # Extract content between <answer> tags if present
    if "<answer>" in out:
        start = out.find("<answer>") + len("<answer>")
        end = out.find("</answer>")
        if end != -1:
            out = out[start:end].strip()
    # Remove <thought> section if only <thought> exists
    elif "<thought>" in out:
        start = out.find("</thought>") + len("</thought>")
        out = out[start:].strip()
    
    # Debug: print extracted content
    print("=== DEBUG output_postprocess_func ===")
    print("Raw output:", output)
    print("Extracted:", repr(out))
    print("=============")
    
    return out

# Initialize evaluator
evaluator = Evaluator(
    llm=llm,
    collate_func=collate_func,
    output_postprocess_func=output_postprocess_func,
    verbose=True,
    num_workers=1  # Reduce concurrency to avoid API limits
)

# ========== Improved F1-chain and F1-graph calculation functions ==========
def f1_chain(prediction: str, label: str) -> float:
    from evoagentx.benchmark.measures import f1_score
    return f1_score(prediction, label)

def f1_graph(prediction: str, label: str) -> float:
    def parse_graph_improved(text):
        """Improved graph parsing function for accurate node and edge extraction"""
        import re
        nodes = []
        edges = []
        
        # Normalize text
        text = text.strip()
        lines = text.splitlines()
        
        node_section = False
        edge_section = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect node section
            if line.lower().startswith("node:"):
                node_section = True
                edge_section = False
                continue
            
            # Detect edge section
            if line.lower().startswith("edge:"):
                edge_section = True
                node_section = False
                continue
            
            # Parse nodes
            if node_section:
                # Match "number: content" format
                node_match = re.match(r'^(\d+):\s*(.+)$', line)
                if node_match:
                    node_num = node_match.group(1)
                    node_content = node_match.group(2).strip()
                    nodes.append(f"{node_num}: {node_content}")
            
            # Parse edges
            if edge_section:
                # Extract all edges within parentheses
                edge_matches = re.findall(r'\(([^)]+)\)', line)
                for edge_match in edge_matches:
                    edge = edge_match.strip()
                    if edge and ',' in edge:
                        edges.append(f"({edge})")
        
        return set(nodes), set(edges)
    
    def normalize_text(text):
        """Normalize text for better matching accuracy"""
        import re
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        # Standardize punctuation
        text = text.replace(',', ',').replace('.', '.').replace(':', ':')
        return text.strip().lower()
    
    def semantic_similarity(text1, text2):
        """Calculate semantic similarity"""
        text1_norm = normalize_text(text1)
        text2_norm = normalize_text(text2)
        
        # Simple word overlap calculation
        words1 = set(text1_norm.split())
        words2 = set(text2_norm.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def build_graph_structure(nodes, edges):
        """Build graph structure and calculate topological features"""
        import re
        from collections import defaultdict, deque
        
        # Parse node IDs
        node_ids = set()
        node_content_map = {}
        for node in nodes:
            match = re.match(r'^(\d+):\s*(.+)$', node)
            if match:
                node_id = match.group(1)
                content = match.group(2)
                node_ids.add(node_id)
                node_content_map[node_id] = content
        
        # Build adjacency list
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        for edge in edges:
            match = re.match(r'\(([^,]+),\s*([^)]+)\)', edge)
            if match:
                from_node = match.group(1).strip()
                to_node = match.group(2).strip()
                
                # Handle START and END nodes
                if from_node == "START":
                    from_node = "0"
                if to_node == "END":
                    to_node = str(max(int(n) for n in node_ids) + 1 if node_ids else 1)
                
                if from_node in node_ids and to_node in node_ids:
                    graph[from_node].append(to_node)
                    in_degree[to_node] += 1
        
        # Calculate topological features
        def get_topological_features():
            """Calculate topological sorting and features"""
            # Topological sort
            queue = deque([node for node in node_ids if in_degree[node] == 0])
            topo_order = []
            visited = set()
            
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                topo_order.append(node)
                
                for neighbor in graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
            
            # Calculate features
            features = {
                'node_count': len(node_ids),
                'edge_count': len(edges),
                'max_depth': len(topo_order),
                'avg_branching': sum(len(graph[node]) for node in node_ids) / len(node_ids) if node_ids else 0,
                'parallel_paths': sum(1 for node in node_ids if len(graph[node]) > 1),
                'sequential_paths': sum(1 for node in node_ids if len(graph[node]) == 1)
            }
            
            return features, topo_order
        
        return get_topological_features()
    
    def structural_similarity(pred_nodes, pred_edges, label_nodes, label_edges):
        """Calculate graph structure similarity"""
        try:
            pred_features, pred_topo = build_graph_structure(pred_nodes, pred_edges)
            label_features, label_topo = build_graph_structure(label_nodes, label_edges)
            
            # Calculate feature similarity
            feature_similarity = 0
            total_features = 0
            
            for key in pred_features:
                if key in label_features:
                    pred_val = pred_features[key]
                    label_val = label_features[key]
                    
                    if pred_val == 0 and label_val == 0:
                        similarity = 1.0
                    elif pred_val == 0 or label_val == 0:
                        similarity = 0.0
                    else:
                        similarity = min(pred_val, label_val) / max(pred_val, label_val)
                    
                    feature_similarity += similarity
                    total_features += 1
            
            avg_feature_similarity = feature_similarity / total_features if total_features > 0 else 0.0
            
            # Calculate topological order similarity
            topo_similarity = 0
            if pred_topo and label_topo:
                common_nodes = set(pred_topo) & set(label_topo)
                if common_nodes:
                    pred_positions = {node: i for i, node in enumerate(pred_topo)}
                    label_positions = {node: i for i, node in enumerate(label_topo)}
                    
                    position_diffs = []
                    for node in common_nodes:
                        diff = abs(pred_positions[node] - label_positions[node])
                        position_diffs.append(diff)
                    
                    if position_diffs:
                        avg_diff = sum(position_diffs) / len(position_diffs)
                        max_possible_diff = max(len(pred_topo), len(label_topo))
                        topo_similarity = 1.0 - (avg_diff / max_possible_diff)
            
            return 0.6 * avg_feature_similarity + 0.4 * topo_similarity
            
        except Exception as e:
            print(f"Error calculating structural similarity: {e}")
            return 0.0
    
    def improved_f1(set_pred, set_label, similarity_threshold=0.7):
        """Improved F1 calculation considering semantic similarity"""
        if not set_pred or not set_label:
            return 0.0
        
        # Calculate exact matches
        exact_matches = len(set_pred & set_label)
        
        # Calculate semantic matches
        semantic_matches = 0
        for pred_item in set_pred:
            if pred_item in set_label:
                continue  # Already exact match
            for label_item in set_label:
                if label_item in set_pred:
                    continue  # Already exact match
                if semantic_similarity(pred_item, label_item) >= similarity_threshold:
                    semantic_matches += 1
                    break
        
        total_matches = exact_matches + semantic_matches
        precision = total_matches / len(set_pred) if set_pred else 0.0
        recall = total_matches / len(set_label) if set_label else 0.0
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Parse graph structure
    pred_nodes, pred_edges = parse_graph_improved(prediction)
    label_nodes, label_edges = parse_graph_improved(label)
    
    # Calculate improved F1 score
    node_f1 = improved_f1(pred_nodes, label_nodes)
    edge_f1 = improved_f1(pred_edges, label_edges)
    
    # Calculate structural similarity
    structural_sim = structural_similarity(pred_nodes, pred_edges, label_nodes, label_edges)
    
    # Comprehensive score: combine semantic and structural matching
    semantic_score = 0.6 * node_f1 + 0.4 * edge_f1
    final_score = 0.7 * semantic_score + 0.3 * structural_sim
    
    return final_score

# Run evaluation
with suppress_logger_info():
    try:
        results = evaluator.evaluate(
            graph=workflow,
            benchmark=benchmark,
            eval_mode="train",  # Match mode
            sample_k=10
        )
        
        # Print prediction, label, and F1-chain/F1-graph for each sample
        f1_chain_scores = []
        f1_graph_scores = []
        successful_evaluations = 0
        
        for i, example in enumerate(benchmark.get_train_data(sample_k=10)):
            try:
                record = evaluator.get_example_evaluation_record(benchmark, example)
                if record:
                    pred = record["prediction"]
                    label = record["label"]
                    print(f"\n--- Sample {i+1} ---")
                    print("Ground truth label:", label)
                    print("Model prediction:", pred)
                    chain_score = f1_chain(pred, label)
                    graph_score = f1_graph(pred, label)
                    print("F1-chain:", chain_score)
                    print("F1-graph:", graph_score)
                    f1_chain_scores.append(chain_score)
                    f1_graph_scores.append(graph_score)
                    successful_evaluations += 1
                else:
                    print(f"\n--- Sample {i+1} evaluation failed ---")
            except Exception as e:
                print(f"\n--- Sample {i+1} processing failed: {str(e)} ---")
        
        print(f"\nSuccessful evaluations: {successful_evaluations}/10")
        
        if f1_chain_scores:
            print("Average F1-chain:", sum(f1_chain_scores)/len(f1_chain_scores))
        if f1_graph_scores:
            print("Average F1-graph:", sum(f1_graph_scores)/len(f1_graph_scores))

        # Print raw content of one sample's ground truth label
        example = benchmark.get_train_data(sample_k=1)[0]
        label = benchmark._get_label(example)
        print("Ground truth label raw content:\n", label)

        # Output evaluation results
        print("Evaluation metrics: ", results)

        # ====== Batch generate official format pred.json ======
        import json
        import re
        input_file = "data/worfbench/worfbench_test.json"
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        # Print content of benchmark.get_train_data(sample_k=1)[0]
        print("benchmark.get_train_data(sample_k=1)[0]:", benchmark.get_train_data(sample_k=1)[0])

        def parse_workflow(llm_output):
            import re
            nodes = []
            edges = []
            node_section = False
            edge_section = False
            for line in llm_output.splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("node:"):
                    node_section = True
                    edge_section = False
                    continue
                if line.lower().startswith("edge:"):
                    edge_section = True
                    node_section = False
                    # Parse all edges in this line
                    for em in re.findall(r'\(([^)]+)\)', line):
                        if ',' in em:
                            from_idx, to_idx = [x.strip() for x in em.split(',', 1)]
                            edges.append([from_idx, to_idx])
                    continue
                if node_section:
                    m = re.match(r'^\s*(\d+)[\.:]\s*(.+)$', gode = line)
                    if m:
                        nodes.append(m.group(2).strip())
                if edge_section:
                    # Parse additional edge lines if any
                    for em in re.findall(r'\(([^)]+)\)', line):
                        if ',' in em:
                            from_idx, to_idx = [x.strip() for x in em.split(',', 1)]
                            edges.append([from_idx, to_idx])
            return nodes, edges

        # Sample 50 items based on gold.json
        import random
        with open('WorfBench/gold.json', 'r', encoding='utf-8') as f:
            gold_full = json.load(f)
        sample_k = 50
        gold_50 = random.sample(gold_full, sample_k)
        gold_ids = [item['id'] for item in gold_50]
        with open('gold_50.json', 'w', encoding='utf-8') as f:
            json.dump(gold_50, f, ensure_ascii=False, indent=2)
        print(f"Sampled gold_50.json, count: {len(gold_50)}")

        # Find corresponding samples in input_data using gold_50.json IDs
        input_map = {item['id']: item for item in input_data}
        pred_records = []
        for idx, gold_item in enumerate(gold_50):
            item = input_map.get(gold_item['id'])
            if not item:
                print(f"Input sample with id={gold_item['id']} not found, skipping")
                nodes, edges = [], []
            else:
                example = {
                    "messages": item["conversations"],
                    "source": item.get("source", "")
                }
                print(f"\nProcessing sample {idx+1}: id={item['id']}")
                try:
                    prompt = collate_func(example)["problem"]
                    output = llm.generate(prompt=prompt)
                    llm_output = output_postprocess_func(output)
                    print(f"LLM prediction raw content: {repr(llm_output)}")
                    nodes, edges = parse_workflow(llm_output)
                    print(f"Parsed result nodes: {nodes}, edges: {edges}\n")
                except Exception as e:
                    print(f"Exception: {e}")
                    import traceback
                    traceback.print_exc()
                    nodes, edges = [], []
            pred_records.append({
                "id": gold_item["id"],
                "workflow": {
                    "nodes": nodes,
                    "edges": edges
                }
            })
        with open('pred.json', 'w', encoding='utf-8') as f:
            json.dump(pred_records, f, ensure_ascii=False, indent=2)
        print("Generated pred.json for official evaluation!")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()