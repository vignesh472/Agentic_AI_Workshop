# agents/prerequisite_detector/graph.py

# A basic concept dependency map (extend this dynamically later)
CONCEPT_DEPENDENCY_GRAPH = {
    "Recursion": ["Functions", "Loops"],
    "Linked Lists": ["Pointers", "Recursion"],
    "Trees": ["Recursion", "Linked Lists"],
    "Graphs": ["Trees", "Hash Tables"],
    "Hash Tables": ["Arrays"],
    "Loops": ["Variables"],
    "Functions": ["Variables"],
    "Pointers": ["Memory Basics"]
}
