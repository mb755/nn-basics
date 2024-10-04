import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import networkx as nx
from layers import Dense, Softmax


def visualize_2D_classification_data(dataset):
    X = dataset.X
    y = dataset.y
    """Visualize a 2D classification dataset."""
    plt.scatter(X[:, 0], X[:, 1], c=np.argmax(y, axis=1), cmap="viridis")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("2D Classification Dataset")
    plt.colorbar(label="Class")
    plt.show()


def generate_neuron_image(network, layer_index, neuron_index):
    xx, yy = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    current_output = grid
    for i in range(layer_index):
        current_output = network.layers[i].forward(current_output)

    neuron_output = current_output[:, neuron_index]

    fig, ax = plt.subplots(figsize=(1, 1), dpi=100)
    ax.imshow(neuron_output.reshape(100, 100), extent=(-3, 3, -3, 3), cmap="viridis")
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    image = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    plt.close(fig)
    return image


# CR TODO: split out the output visualization into a separate function
def visualize_network(model, dataset):
    X = dataset.X
    y_onehot = dataset.y
    G = nx.DiGraph()

    layer_sizes = [X.shape[1]]
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer_sizes.append(layer.output_size)
        elif isinstance(layer, Softmax):
            layer_sizes.append(layer_sizes[-1])

    pos = {}
    for i, size in enumerate(layer_sizes):
        for j in range(size):
            G.add_node(f"L{i}_{j}")
            pos[f"L{i}_{j}"] = (i, j - size / 2 + 0.5)

    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                G.add_edge(f"L{i}_{j}", f"L{i+1}_{k}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax1,
        arrows=True,
        arrowstyle="-",
        min_source_margin=15,
        min_target_margin=15,
    )

    icon_size = (ax1.get_xlim()[1] - ax1.get_xlim()[0]) * 0.05

    for n in G.nodes:

        parts = n[1:].split("_")
        if len(parts) < 2:  # Skip nodes that don't follow the L{layer}_{neuron} format
            continue
        layer, neuron = int(parts[0]), int(parts[1])

        x, y = pos[n]

        if layer == 0:  # Input layer
            image = np.ones((100, 100, 3)) * 0.8  # Light gray for input
        else:
            image = generate_neuron_image(model, layer, neuron)

        # Calculate the extent of the image
        extent = [
            x - icon_size / 2,
            x + icon_size / 2,
            y - icon_size / 2,
            y + icon_size / 2,
        ]

        # Display the image
        ax1.imshow(
            image, extent=extent, zorder=2
        )  # zorder=2 to ensure it's above the edges

    ax1.set_title("Neural Network Architecture")
    ax1.set_xlim(
        min(pos.values(), key=lambda x: x[0])[0] - 0.5,
        max(pos.values(), key=lambda x: x[0])[0] + 0.5,
    )
    ax1.set_ylim(
        min(pos.values(), key=lambda x: x[1])[1] - 0.5,
        max(pos.values(), key=lambda x: x[1])[1] + 0.5,
    )
    ax1.axis("off")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.argmax(Z, axis=1).reshape(xx.shape)

    ax2.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlBu")
    scatter = ax2.scatter(
        X[:, 0],
        X[:, 1],
        c=np.argmax(y_onehot, axis=1),
        edgecolor="black",
        cmap="RdYlBu",
    )
    ax2.set_xlabel("X1")
    ax2.set_ylabel("X2")
    ax2.set_title("Decision Boundary")

    plt.colorbar(scatter, ax=ax2)
    plt.tight_layout()
    plt.show()
