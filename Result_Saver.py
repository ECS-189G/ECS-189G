'''
Concrete ResultModule class for a specific experiment ResultModule output
'''


from local_code.base_class.result import result
import pickle


class Result_Saver(result):
    data = None
    result_destination_folder_path = None
    result_destination_file_name = None
    
    def save(self):
        print('saving results...')
        f = open(self.result_destination_folder_path + self.result_destination_file_name, 'wb')
        pickle.dump(self.data, f)
        f.close()