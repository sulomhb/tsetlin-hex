import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.graphs import Graphs
import random

# Default arguments for different board sizes
def defaultArgs(**kwargs):
    args = argparse.Namespace()
    args.board_size = 11
    args.epochs = 25
    args.number_of_clauses = 4500
    args.T = 6000
    args.s = 0.8
    args.depth = 5
    args.hypervector_size = 512
    args.hypervector_bits = 2
    args.message_size = 512
    args.message_bits = 2
    args.max_included_literals = 16
    args.double_hashing = False
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args

# Helper function to encode a board state
def encode_board_state(board):
    return np.array([1 if cell == 'X' else -1 if cell == 'O' else 0 for cell in board.flatten()])

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

    def is_fully_expanded(self):
        return len(self.children) >= len(get_legal_moves(self.state))

    def expand(self):
        unvisited_moves = [move for move in get_legal_moves(self.state) if move not in [child.state for child in self.children]]
        if unvisited_moves:
            move = random.choice(unvisited_moves)
            new_state = apply_move(self.state, move)
            child_node = Node(new_state, parent=self)
            self.children.append(child_node)
            return child_node
        return None

    def best_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda child: child.value / (child.visits + 1e-6) + exploration_weight * np.sqrt(np.log(self.visits + 1) / (child.visits + 1e-6)))

def monte_carlo_tree_search(root, tm, simulations=1000):
    for _ in range(simulations):
        node = root
        while node.children and node.is_fully_expanded():
            node = node.best_child()
        if not node.is_fully_expanded():
            node = node.expand()
        result = simulate(node.state, tm)
        backpropagate(node, result)
    return root.best_child(0)

def simulate(state, tm):
    board_vector = encode_board_state(state)
    prediction = tm.predict([board_vector])[0]
    return 1 if prediction == 1 else 0

def backpropagate(node, result):
    while node:
        node.visits += 1
        node.value += result
        node = node.parent

# Hex-specific helper functions
def get_legal_moves(state):
    return [(i, j) for i, row in enumerate(state) for j, cell in enumerate(row) if cell == '.']

def apply_move(state, move):
    new_state = state.copy()
    new_state[move] = 'X' if sum(row.count('X') for row in state) <= sum(row.count('O') for row in state) else 'O'
    return new_state

# Initialize graphs for the TM
def createGraphs(X_data, nodeNames, nEdgesList, edges, args, boardSize, forTraining=True, baseGraphs=None):
    graphs = Graphs(
        number_of_graphs=len(X_data),
        symbols=["O", "X", "."],
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing
    ) if forTraining else Graphs(len(X_data), init_with=baseGraphs)

    for graphId, board in enumerate(X_data):
        for nodeId, nodeName in enumerate(nodeNames):
            graphs.add_graph_node(graphId, nodeName, nEdgesList[nodeId])
            sym = board[nodeId]
            graphs.add_graph_node_property(graphId, nodeName, sym)

    graphs.encode()
    return graphs

# Plotting functions
def plotConfusionMatrix(yTrue, yPred, boardSize):
    cm = confusion_matrix(yTrue, yPred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
    plt.title(f"Confusion Matrix (Board Size {boardSize})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(f"confusion_matrix_{boardSize}.png")
    plt.close()

def plotClauseWeightDistribution(tm):
    weights = tm.get_state()[1]
    posWeights = weights[:tm.number_of_clauses]
    plt.figure()
    plt.hist(posWeights, bins=20, color='purple', edgecolor='black')
    plt.title("Distribution of Clause Weights")
    plt.xlabel("Clause Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("clause_weights_distribution.png")
    plt.close()

# Main function
if __name__ == "__main__":
    args = defaultArgs()

    # Load and preprocess data
    data = pd.read_csv("hex_game_data_complete.csv", dtype=str)
    boardSize = args.board_size
    nodeNames = [f"{i}_{j}" for i in range(boardSize) for j in range(boardSize)]
    X = data[nodeNames].values
    y = data["Winner"].values.astype(int)

    splitIdx = int(len(data) * 0.9)
    X_train, X_test = X[:splitIdx], X[splitIdx:]
    y_train, y_test = y[:splitIdx], y[splitIdx:]

    # Initialize TM
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses, args.T, args.s,
        depth=args.depth, message_size=args.message_size,
        message_bits=args.message_bits, max_included_literals=args.max_included_literals
    )

    # Training the TM
    for epoch in range(args.epochs):
        tm.fit(createGraphs(X_train, nodeNames, [], [], args, boardSize), y_train, epochs=1, incremental=True)

    # MCTS Integration
    initial_state = np.array([['.'] * boardSize] * boardSize)
    root = Node(initial_state)
    best_move_node = monte_carlo_tree_search(root, tm, simulations=1000)

    print(f"Best move state:\n{best_move_node.state}")
