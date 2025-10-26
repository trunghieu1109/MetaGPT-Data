import json
from examples.generate_sample_data.graph_based_scenario.constants import MAX_CUSTOM_OP, MAX_REVISE_OP, MAX_LOOP
import random
from collections import Counter

class Node:
    def __init__(self, name, label=None):
        self.name = name
        self.label = label or name
        self.out_edges = set()

    def __repr__(self):
        return f"Node({self.name}, label='{self.label}')"


class Edge:
    def __init__(self, start, end, constraint=None):
        self.start = start
        self.end = end
        self.constraint = constraint  # Ví dụ: "condition=True" hoặc "weight=2"

    def __repr__(self):
        return f"Edge({self.start} -> {self.end}, constraint={self.constraint})"

operators_set = ['Custom', 'ScEnsemble', 'Review', 'Revise', 'Format', 'Debater', 'Judge', 'Programmer', 'CustomCodeGenerate', 'Test']

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, name, label=None):
        """Thêm node vào đồ thị"""
        if name not in self.nodes:
            node = Node(name, label)
            self.nodes[name] = node
        return self.nodes[name]

    def add_edge(self, start, end, constraint=None):
        """Thêm cạnh giữa hai node, có thể có ràng buộc"""
        if start not in self.nodes or end not in self.nodes:
            raise ValueError("Both nodes must exist before adding an edge.")
        edge = Edge(start, end, constraint)
        self.edges.append(edge)
        self.nodes[start].out_edges.add(edge)

    def find_paths(self, start, end, path=None, visited=None, curr_len = 0, num_cag = 0, num_agc = 0, max_lens = 25):
        """Tìm tất cả các đường đi hợp lệ từ start đến end"""
        if path is None:
            path = [self.nodes[start].label]
        if visited is None:
            visited = set()

        if start == end:
            oc = 0
            for node in path:
                if node == 'Start Loop':
                    oc += 1
                elif node == 'End Loop':
                    oc -= 1
                    
                if oc < 0:
                    return
                
            if oc != 0:
                return
            
            yield path
            return
        
        if curr_len > max_lens:
            return

        visited.add(start)
        for edge in self.nodes[start].out_edges:
            next_node = edge.end
            node_label = self.nodes[next_node].label
            is_valid = False
            
            oc = 0
            count_loop = 0
            count_custom = 0
            count_revise = 0
            for node in path:
                if node == 'Start Loop':
                    oc += 1
                elif node == 'End Loop':
                    oc -= 1
                    if oc >= 0:
                        count_loop += 1
                elif node == 'Custom':
                    count_custom += 1
                elif node == 'Revise':
                    count_revise += 1
                    
                if oc < 0:
                    return
            
            if count_loop > MAX_LOOP:
                return
            
            if count_custom > MAX_CUSTOM_OP:
                return
            
            if count_revise > MAX_REVISE_OP:
                return
            
            if node_label == 'End Loop':
                for node in reversed(path):
                    if node == 'Start Loop':
                        is_valid = True
                        break
                    elif node == 'End Loop':
                        break
            elif node_label == 'Review':
                if any(item in path for item in ['Custom', 'ScEnsemble']):
                    is_valid = True
            elif node_label == 'Start Loop':
                is_valid = True
                for node in reversed(path):
                    if node == 'Start Loop':
                        is_valid = False
                        break
                    elif node == 'End Loop':
                        is_valid = True
                        break
            else:
                is_valid = True
                    
            if not is_valid:
                continue
                
            yield from self.find_paths(next_node, end, path + [self.nodes[next_node].label], visited.copy(), curr_len + int(node_label in operators_set), num_cag, num_agc, max_lens)

    def __repr__(self):
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)})"
    
    def to_dict(self):
        return {
            'nodes': [v for k, v in self.nodes.items()],
            'edges': self.edges
        }


# # ------------------------------
# # Khởi tạo đồ thị theo mô tả
# # ------------------------------
# graph = Graph()

# # Tạo node
# # for n, label in [
# #     ("A", "Custom"), ("B", "AnswerGenerate"), ("C", "ScEnsemble"),
# #     ("D", "Review"), ("E", "Revise"), ("F", "Format"),
# #     ("G", "Debater"), ("I", "Start Loop"), ("J", "End Loop"),
# #     ("P", "Start"), ("Q", "End")
# # ]:
# #     graph.add_node(n, label)

# for n, label in [
#     ("A", "Custom"), ("C", "ScEnsemble"),
#     ("D", "Review"), ("E", "Revise"), ("F", "Format"),
#     ("G", "Debater"), ("I", "Start Loop"), ("J", "End Loop"),
#     ("P", "Start"), ("Q", "End")
# ]:
#     graph.add_node(n, label)

# # Tạo cạnh
# # edges = [
# #     ("P", "A"), ("P", "B"), ("P", "I"),
# #     ("A", "B"), ("A", "A"), ("A", "D"), ("A", "F"), ("A", "I"), ("A", "J"),
# #     ("I", "A"), ("B", "A"), ("C", "A"), ("J", "A"),
# #     ("B", "D"), ("B", "F"), ("B", "I"), ("B", "J"), ("C", "B"), ("I", "B"), ("J", "B"),
# #     ("J", "C"), ("C", "D"), ("C", "F"), ("C", "I"),
# #     ("D", "E"), ("I", "D"),
# #     ("E", "J"), ("E", "F"),
# #     ("J", "F"), ("I", "G"), ("G", "J"), ("J", "D"),
# #     ("A", "Q"), ("B", "Q"), ("C", "Q"), ("E", "Q"), ("F", "Q")
# # ]

# edges = [
#     ("P", "A"), ("P", "I"),
#     ("A", "A"), ("A", "D"), ("A", "F"), ("A", "I"), ("A", "J"),
#     ("I", "A"), ("C", "A"), ("J", "A"),
#     ("J", "C"), ("C", "D"), ("C", "F"), ("C", "I"),
#     ("D", "E"), ("I", "D"),
#     ("E", "J"), ("E", "F"),
#     ("J", "F"), ("I", "G"), ("G", "J"), ("J", "D"),
#     ("A", "Q"), ("C", "Q"), ("E", "Q"), ("F", "Q")
# ]

# for start, end in edges:
#     graph.add_edge(start, end)

# # ------------------------------
# # Tìm các đường đi hợp lệ
# # ------------------------------
# print("Graph summary:", graph)
# # print(graph.to_dict())
# print("\nAll possible paths from Start (P) to End (Q):")

# paths = []
# max_lens = 0
# count_format_mas = 0
# formatted_paths = []
# op_counts = []
# for path in graph.find_paths("P", "Q", curr_len=0, num_cag = 0, num_agc = 0, max_lens = 16):
#     if 'Format' in path:
#         formatted_paths.append(" -> ".join(path))
#     else:
#         paths.append(" -> ".join(path))
#     count_op = 0
#     for p in path:
#         if p in operators_set:
#             count_op += 1
#     op_counts.append(count_op)
    

# length_counts = Counter(op_counts)

# for x in range(1, 21):
#     print(f"Length {x}: {length_counts.get(x, 0)} paths")
    
# formatted_paths = random.sample(formatted_paths, 1000)
# paths = paths + formatted_paths

# print(len(paths))

# # for path in paths[:10]:
# #     print("--------------------------")
# #     print(path)
    