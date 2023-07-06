import os
import shutil

def empty_directory(remove_directries_list):
    for remove_directory in remove_directries_list:
        if os.path.isdir(remove_directory):
            shutil.rmtree(remove_directory)
            os.mkdir(remove_directory)
        else:
            os.mkdir(remove_directory)

def main():
    '''
    remove_directries = [
        'output-python/T20230411', 'train_dataset', 'test_dataset', 'sample_dataset', 
        'dataset_all', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5'
        ]
    '''
    remove_directries = [
        'train_dataset', 'test_dataset', 'sample_dataset', 
        'dataset_all', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5'
        ]
    empty_directory(remove_directries)

if __name__ == "__main__":
    main()
