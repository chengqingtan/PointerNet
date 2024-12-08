import torch
import matplotlib.pyplot as plt
from PointerNet import PointerNet
from Data_Generator import TSPDataset

def predict_and_visualize(model_path, embedding_size, hiddens, nof_lstms, dropout, bidir, num_points=5, use_cuda=False):
    """Load model and visualize predictions on TSP problem."""
    model = PointerNet(embedding_size, hiddens, nof_lstms, dropout, bidir)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    if use_cuda and torch.cuda.is_available():
        model.cuda()

    # Generate a single TSP instance
    test_dataset = TSPDataset(1, num_points)
    sample = test_dataset[0]
    points = sample['Points'].unsqueeze(0)  # Add batch dimension

    if use_cuda and torch.cuda.is_available():
        points = points.cuda()

    with torch.no_grad():
        _, predictions = model(points)

    # Extract the coordinates and predicted order
    points = points.squeeze(0).cpu().numpy()
    predicted_order = predictions.squeeze(0).cpu().numpy()

    # Visualize
    plt.figure(figsize=(6, 6))
    plt.scatter(points[:, 0], points[:, 1], color='blue', label='Points')

    # Plot the path
    for i in range(len(predicted_order) - 1):
        start = points[predicted_order[i]]
        end = points[predicted_order[i + 1]]
        plt.plot([start[0], end[0]], [start[1], end[1]], color='red')

    # Connect the last point to the first
    start = points[predicted_order[-1]]
    end = points[predicted_order[0]]
    plt.plot([start[0], end[0]], [start[1], end[1]], color='red', linestyle='dashed')

    plt.title("TSP Prediction")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    predict_and_visualize(
        model_path="./saved/pointer_net_model_5.pt",
        embedding_size=128,
        hiddens=512,
        nof_lstms=2,
        dropout=0.0,
        bidir=True,
        num_points=5,
        use_cuda=True
    )
