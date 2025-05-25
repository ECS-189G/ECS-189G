from local_code.stage_4_code.Data_Clean_Classification import Clean_Reviews
import string

if 1:
    # Use Windows paths
    train_file_path = r'C:\Users\Sumana\UCD\ECS189G_Winter_2025_Source_Code_Template\data\stage_4_data\text_classification\train'
    test_file_path = r'C:\Users\Sumana\UCD\ECS189G_Winter_2025_Source_Code_Template\data\stage_4_data\text_classification\test'

    print("Processing train dataset")
    clean_train = Clean_Reviews(train_file_path)
    clean_train.clean_and_save('clean_reviews_train.json')

    print("Processing test dataset")
    clean_test = Clean_Reviews(test_file_path)
    clean_test.clean_and_save('clean_reviews_test.json')

    print("Done cleaning")
