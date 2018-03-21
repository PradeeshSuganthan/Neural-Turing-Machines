from keras.preprocessing import text
import glob

def import_files(text_files_location="./shakespeare/*.txt"):
    text_file_names = glob.glob(text_files_location)
    text_list = []
    for text_file_name in text_file_names:
        text_file = open(text_file_name)
        text_list.append(text_file.read())
        text_file.close()
     return text_list
    
    
import_files()