import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from time import time
from sklearn.metrics import confusion_matrix, accuracy_score
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine
from GraphTsetlinMachine.graphs import Graphs

def defaultArgs(**kwargs):
    """
    Set default arguments and adjust parameters based on board size.
    Different board complexities (e.g., 3x3, 7x7, 11x11).
    """
    epochs = 25
    boardSize = 11
    depth = 5
    maxIncludedLiterals = 16

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=epochs, type=int)
    parser.add_argument("--board_size", default=boardSize, type=int)
    parser.add_argument("--depth", default=depth, type=int)
    parser.add_argument("--hypervector_size", default=512, type=int)
    parser.add_argument("--hypervector_bits", default=2, type=int)
    parser.add_argument("--message_size", default=512, type=int)
    parser.add_argument("--message_bits", default=2, type=int)
    parser.add_argument('--double_hashing', dest='double_hashing', default=False, action='store_true')
    parser.add_argument("--max_included_literals", default=maxIncludedLiterals, type=int)

    args = parser.parse_args(args=[])

    # Adjust TM parameters based on board size.
    if args.board_size == 3:
        args.number_of_clauses = 200
        args.T = 400
        args.s = 1.2
    elif args.board_size == 5:
        args.number_of_clauses = 1000
        args.T = 800
        args.s = 1.0
    elif args.board_size == 7:
        args.number_of_clauses = 1500
        args.T = 2000
        args.s = 0.9
    elif args.board_size == 9:
        args.number_of_clauses = 3200
        args.T = 4000
        args.s = 0.8
    elif args.board_size == 11:
        args.number_of_clauses = 4500
        args.T = 6000
        args.s = 0.8

    for key, value in kwargs.items():
        if hasattr(args, key):
            setattr(args, key, value)
    
    return args

def positionToEdgeId(pos, boardSize):
    return pos[0] * boardSize + pos[1]

def createGraphs(X_data, nodeNames, nEdgesList, edges, args, boardSize, forTraining=True, baseGraphs=None):
    """
    Create Graph objects for training or testing data, encoding the Hex board states.
    """
    if forTraining:
        graphs = Graphs(
            number_of_graphs=len(X_data),
            symbols=["O", "X", "."],
            hypervector_size=args.hypervector_size,
            hypervector_bits=args.hypervector_bits,
            double_hashing=args.double_hashing
        )
    else:
        graphs = Graphs(len(X_data), init_with=baseGraphs)

    for graphId in range(X_data.shape[0]):
        graphs.set_number_of_graph_nodes(graphId, boardSize ** 2)

    graphs.prepare_node_configuration()

    for graphId, board in enumerate(X_data):
        for nodeId, nodeName in enumerate(nodeNames):
            graphs.add_graph_node(graphId, nodeName, nEdgesList[nodeId])
            sym = board[nodeId]
            graphs.add_graph_node_property(graphId, nodeName, sym)

    graphs.prepare_edge_configuration()

    for graphId in range(X_data.shape[0]):
        for edge in edges:
            srcId = positionToEdgeId(edge[0], boardSize)
            destId = positionToEdgeId(edge[1], boardSize)
            graphs.add_graph_node_edge(graphId, nodeNames[srcId], nodeNames[destId], edge_type_name="Plain")
            graphs.add_graph_node_edge(graphId, nodeNames[destId], nodeNames[srcId], edge_type_name="Plain")

    graphs.encode()
    return graphs

def plotAccuracies(trainAccuracies, testAccuracies):
    """
    Plot training and testing accuracies over epochs.
    This visual aid supports the methodology's emphasis on monitoring performance 
    throughout training and identifying where performance plateaus.
    """
    plt.figure()
    plt.plot(range(1, len(trainAccuracies)+1), [a*100 for a in trainAccuracies], marker='o', label='Train Accuracy')
    plt.plot(range(1, len(testAccuracies)+1), [a*100 for a in testAccuracies], marker='o', label='Test Accuracy')
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()
    plt.savefig("accuracy_epochs.png", dpi=300)
    plt.close()

def plotConfusionMatrix(yTrue, yPred, boardSize):
    """
    Plot a confusion matrix to visualize classification performance.
    """
    cm = confusion_matrix(yTrue, yPred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Loss", "Win"], yticklabels=["Loss", "Win"])
    plt.title(f"Confusion Matrix (Board Size {boardSize})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

def plotClauseWeightDistribution(tm):
    """
    Plot the distribution of clause weights for interpretability assessment.
    """
    weights = tm.get_state()[1]
    posWeights = weights[0:tm.number_of_clauses]  # Assuming clause weights are structured this way.
    plt.figure()
    plt.hist(posWeights, bins=20, color='purple', edgecolor='black')
    plt.title("Distribution of Clause Weights")
    plt.xlabel("Clause Weight")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("clause_weights_distribution.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Parse arguments and set defaults
    args = defaultArgs()

    # Load data and board configuration
    data = pd.read_csv("hex_game_data_complete.csv", dtype=str)
    boardSize = args.board_size
    nodeNames = [f"{i}_{j}" for i in range(1, boardSize + 1) for j in range(1, boardSize + 1)]
    X = data[nodeNames].values
    y = data["Winner"].values.astype(int)

    # Split data into training and test sets (90-10 split)
    splitIdx = int(len(data) * 0.9)
    X_train, X_test = X[:splitIdx], X[splitIdx:]
    y_train, y_test = y[:splitIdx], y[splitIdx:]

    # Print label distribution for sanity check
    unique, counts = np.unique(y, return_counts=True)
    labelDistTrain = dict(zip(unique, counts))
    totalTrain = sum(counts)
    print("Overall dataset label distribution:")
    print(f" {labelDistTrain}")
    print(f" Percentage of '0's: {(labelDistTrain.get(0, 0) / totalTrain) * 100:.2f}%")
    print(f" Percentage of '1's: {(labelDistTrain.get(1, 0) / totalTrain) * 100:.2f}%")

    uniqueTest, countsTest = np.unique(y_test, return_counts=True)
    labelDistTest = dict(zip(uniqueTest, countsTest))
    totalTest = sum(countsTest)
    print("Test set label distribution:")
    print(f" {labelDistTest}")
    print(f" Test: Percentage of '0's: {(labelDistTest.get(0, 0) / totalTest) * 100:.2f}%")
    print(f" Test: Percentage of '1's: {(labelDistTest.get(1, 0) / totalTest) * 100:.2f}%")

    # Define edges for the Hex graph structure
    edges = []
    for i in range(boardSize):
        for j in range(boardSize):
            if j < boardSize - 1:
                edges.append(((i, j), (i, j + 1)))
            if i < boardSize - 1:
                edges.append(((i, j), (i + 1, j)))
            if i < boardSize - 1 and j > 0:
                edges.append(((i, j), (i + 1, j - 1)))

    # Determine the number of edges per node (nEdgesList)
    nEdgesList = []
    for i in range(boardSize ** 2):
        if i == 0 or i == boardSize ** 2 - 1:
            nEdgesList.append(2)  # corners with 2 neighbors
        elif i == boardSize - 1 or i == boardSize ** 2 - boardSize:
            nEdgesList.append(3)  # other corner-like positions
        elif i // boardSize == 0 or i // boardSize == boardSize - 1:
            nEdgesList.append(4)  # top/bottom edges
        elif i % boardSize == 0 or i % boardSize == boardSize - 1:
            nEdgesList.append(4)  # left/right edges
        else:
            nEdgesList.append(6)  # center nodes

    # Create graph structures for training and test sets
    graphsTrain = createGraphs(X_train, nodeNames, nEdgesList, edges, args, boardSize, forTraining=True)
    graphsTest = createGraphs(X_test, nodeNames, nEdgesList, edges, args, boardSize, forTraining=False, baseGraphs=graphsTrain)

    # Initialize the Tsetlin Machine
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses,
        args.T,
        args.s,
        depth=args.depth,
        message_size=args.message_size,
        message_bits=args.message_bits,
        max_included_literals=args.max_included_literals,
        grid=(16 * 13, 1, 1),
        block=(128, 1, 1)
    )

    startTraining = time()

    trainAccuracies = []
    testAccuracies = []

    # Train for the specified number of epochs
    for epoch in range(args.epochs):
        if len(y_train) != graphsTrain.number_of_graphs:
            raise ValueError(f"Label/graph mismatch: {len(y_train)} labels vs {graphsTrain.number_of_graphs} graphs.")

        tm.fit(graphsTrain, y_train, epochs=1, incremental=True)
        y_train_pred = tm.predict(graphsTrain)
        y_test_pred = tm.predict(graphsTest)

        trainAcc = accuracy_score(y_train, y_train_pred)
        testAcc = accuracy_score(y_test, y_test_pred)

        trainAccuracies.append(trainAcc)
        testAccuracies.append(testAcc)
        print(f"Epoch #{epoch+1} -- Train Acc: {trainAcc:.4f}, Test Acc: {testAcc:.4f}")

    trainingTime = time() - startTraining
    print(f"Training time: {trainingTime:.2f} seconds")
    print(f"Training graphs: {graphsTrain.number_of_graphs}, Nodes per graph: {boardSize ** 2}")
    print(f"Testing graphs: {graphsTest.number_of_graphs}, Nodes per graph: {boardSize ** 2}")

    averageTrainAcc = np.mean(trainAccuracies)
    averageTestAcc = np.mean(testAccuracies)
    bestTrainAcc = np.max(trainAccuracies)
    worstTrainAcc = np.min(trainAccuracies)
    bestTestAcc = np.max(testAccuracies)
    worstTestAcc = np.min(testAccuracies)

    print("\nPerformance Summary:")
    print(f"Average Train Accuracy: {averageTrainAcc * 100:.2f}%")
    print(f"Average Test Accuracy: {averageTestAcc * 100:.2f}%")
    print(f"Best Train Accuracy: {bestTrainAcc * 100:.2f}%")
    print(f"Worst Train Accuracy: {worstTrainAcc * 100:.2f}%")
    print(f"Best Test Accuracy: {bestTestAcc * 100:.2f}%")
    print(f"Worst Test Accuracy: {worstTestAcc * 100:.2f}%")

    # Plot training/testing accuracies over epochs
    plotAccuracies(trainAccuracies, testAccuracies)

    # Plot confusion matrix for the final predictions
    y_final_pred = tm.predict(graphsTest)
    plotConfusionMatrix(y_test, y_final_pred, boardSize)

    # Plot clause weight distribution
    plotClauseWeightDistribution(tm)
