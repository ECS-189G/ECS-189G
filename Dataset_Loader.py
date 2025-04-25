"""
Concrete I/O class that loads a labeled dataset from a comma-separated text file.
Each line begins with the label, followed by feature values.
"""
from local_code.base_class.dataset import dataset as BaseDataset


class Dataset_Loader(BaseDataset):
    """Loads features and labels from a text dataset file."""
    data = None  # Will hold loaded data as {'X': ..., 'y': ...}
    dataset_source_folder_path = None  # Path to directory containing data file
    dataset_source_file_name = None    # Name of the data file

    def __init__(self, dName=None, dDescription=None):
        # Initialize base dataset with optional metadata
        super().__init__(dName, dDescription)

    def load(self):
        """
        Reads the dataset file line by line, parses integers,
        and separates labels (y) from feature vectors (X).

        Returns:
            dict: A mapping with 'X' for feature list and 'y' for labels list.
        """
        print('Loading data from file...')

        X = []  # List to collect feature vectors
        y = []  # List to collect corresponding labels

        # Construct full file path and open for reading
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        f = open(file_path, 'r')

        for line in f:
            line = line.strip('\n')  # Remove newline at end
            elements = [int(item) for item in line.split(',')]  # Parse all values to int

            label = elements[0]      # First value is the label
            features = elements[1:]  # Remaining values are features

            y.append(label)
            X.append(features)

        f.close()  # Close file handle

        # Store into instance and return
        self.data = {'X': X, 'y': y}
        return self.data