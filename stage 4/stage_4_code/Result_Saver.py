'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle
import os  # Import os for path handling and directory creation

class Result_Saver(result):
    def __init__(self, mName, mDescription):
        super().__init__(mName, mDescription)
        self.method_name = mName
        self.data = None
        self.result_destination_folder_path = None
        self.result_destination_file_name = None

    def save(self, data):
        # Store the incoming results
        self.data = data

        # Check that destination path and filename are set
        if self.result_destination_folder_path is None or self.result_destination_file_name is None:
            print(
                f"[{self.method_name}] Error: Result destination folder path or file name is not set. Cannot save results."
            )
            return

        # Inform the user we're about to save
        print(f"[{self.method_name}] Saving results to {self.result_destination_folder_path}{self.result_destination_file_name}...")

        # Make sure the directory exists
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        full_path = os.path.join(self.result_destination_folder_path, self.result_destination_file_name)

        # Attempt to write out the pickled results
        try:
            with open(full_path, 'wb') as f:
                pickle.dump(self.data, f)
            print(f"[{self.method_name}] Results successfully saved to {full_path}")
        except Exception as e:
            print(f"[{self.method_name}] Error saving results: {e}")
