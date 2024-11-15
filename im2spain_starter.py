import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Function to visualize the dataset and PCA-transformed features
def plot_data(train_feats, train_labels):
    """
    Plots the image locations and PCA-transformed features.
    
    Args:
        train_feats (ndarray): Features of the training dataset.
        train_labels (ndarray): Labels (latitude and longitude) of the training dataset.
    """
    # Plot image locations (latitude and longitude)
    plt.scatter(train_labels[:, 1], train_labels[:, 0], marker=".")
    plt.title("Image Locations")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    # Standardize features and reduce to 2 dimensions using PCA
    transformed_feats = StandardScaler().fit_transform(train_feats)
    transformed_feats = PCA(n_components=2).fit_transform(transformed_feats)

    # Plot images by their first two PCA dimensions, colored by longitude
    plt.scatter(
        transformed_feats[:, 0],
        transformed_feats[:, 1],
        c=train_labels[:, 1],
        marker=".",
    )
    plt.colorbar(label="Longitude")
    plt.title("Image Features by Longitude after PCA")
    plt.show()

# Function to perform k-NN grid search for optimal k
def grid_search(train_features, train_labels, test_features, test_labels, is_weighted=False, verbose=True):
    """
    Performs grid search to find the optimal number of neighbors (k) for k-NN regression.
    Also calculates and plots the mean displacement error (MDE) for each k.
    
    Args:
        train_features (ndarray): Features of the training dataset.
        train_labels (ndarray): Labels (latitude and longitude) of the training dataset.
        test_features (ndarray): Features of the test dataset.
        test_labels (ndarray): Labels (latitude and longitude) of the test dataset.
        is_weighted (bool): Whether to use distance-weighted k-NN. Default is False.
        verbose (bool): Whether to print progress and results. Default is True.
    
    Returns:
        float: Minimum mean displacement error (MDE) achieved.
    """
    # Fit k-NN model with training features
    knn = NearestNeighbors(n_neighbors=100).fit(train_features)

    if verbose:
        print(f"Running grid search for k (is_weighted={is_weighted})")

    # Define range of k values to test
    ks = list(range(1, 11)) + [20, 30, 40, 50, 100]
    mean_errors = []

    for k in ks:
        # Get the k nearest neighbors for test features
        distances, indices = knn.kneighbors(test_features, n_neighbors=k)

        errors = []
        for i, nearest in enumerate(indices):
            # True label for the test sample
            y = test_labels[i]

            # Predict the label using the average of neighbors' labels
            predictions = np.average(
                train_labels[nearest],
                axis=0,
                weights=1 / distances[i] if is_weighted else None,  # Use distance weights if specified
            )
            # Compute Euclidean distance between prediction and true label
            e = np.linalg.norm(predictions - y)
            errors.append(e)

        # Compute mean displacement error (MDE) for this value of k
        e = np.mean(errors)
        mean_errors.append(e)
        if verbose:
            print(f"{k}-NN mean displacement error (miles): {e:.1f}")

    # Plot MDE vs. k
    if verbose:
        plt.plot(ks, mean_errors)
        plt.xlabel("k")
        plt.ylabel("Mean Displacement Error (miles)")
        plt.title("Mean Displacement Error (miles) vs. k in kNN")
        plt.show()

    return min(mean_errors)

# Main function to execute the script
def main():
    """
    Main function to load data, visualize features, find nearest neighbors, and
    perform grid search to optimize k-NN regression.
    """
    print("Predicting GPS from CLIP image features\n")

    # Import data from file
    print("Loading Data")
    data = np.load("im2spain_data.npz")

    train_features = data["train_features"]  # Features of training images
    test_features = data["test_features"]    # Features of test images
    train_labels = data["train_labels"]      # Labels (lat, lon) of training images
    test_labels = data["test_labels"]        # Labels (lat, lon) of test images
    train_files = data["train_files"]        # File names of training images
    test_files = data["test_files"]          # File names of test images

    # Print basic dataset information
    print("Train Data Count:", train_features.shape[0])

    # Visualize the dataset
    plot_data(train_features, train_labels)

    # Find the 5 nearest neighbors for a specific test image
    knn = NearestNeighbors(n_neighbors=3).fit(train_features)
    test_image_index = np.where(test_files == "53633239060.jpg")[0][0]  # Find the index of the test image
    test_image_features = test_features[test_image_index].reshape(1, -1)  # Extract features for this test image

    # Get nearest neighbors and their distances
    distances, indices = knn.kneighbors(test_image_features, n_neighbors=5)

    # Print results for the test image
    print(f"Test Image: 53633239060.jpg")
    print(f"Nearest Neighbors (Indices): {indices.flatten()}")
    print(f"Nearest Neighbors (Distances): {distances.flatten()}")

    # Print the locations of the nearest neighbors
    print("Nearest Neighbors' Locations:")
    for neighbor_idx in indices.flatten():
        lat, lon = train_labels[neighbor_idx]
        print(f"Neighbor Index: {neighbor_idx}, Latitude: {lat:.6f}, Longitude: {lon:.6f}")

    # Perform grid search to find optimal k
    grid_search(train_features, train_labels, test_features, test_labels)
    grid_search(train_features, train_labels, test_features, test_labels, is_weighted=True)

if __name__ == "__main__":
    main()
