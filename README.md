# Generating-music-using-deep-learning



#### This is a project to generate some pop music melodies using different neural network structures on midi files dataset.



### Data preprocess

#### I used Event representation. Details in data_preprocess.py. There are two main methods(Encode and Decode). 

    0-87 note_on 
    88-175 note_off 
    176-275 time_shift


### LSTM model

#### I used LSTM model in RNN.py. There is a pretrained weight in weight/RNN_weight.pt. And there is a example in generated_example/test_RNN_001.midi.  The following is the structure of the model：
    #structue:
    # embedding (276,128)
    # lstm(128,256)
    # fc(256,300)
    # relu()
    # fc(300,276)


#### You can use your data to train the model or generate midi files by changing parameters in RNN.py
    #parameter
    If_train = False
    If_generate = True
    If_save_weight = False
    If_load_weight_for_training = True
    If_load_weight_for_generating = True
    Start_midi_file = 'dataset/data/train/235.midi'    # use first 16 tokens to generate
    Generated_midi_length = 1000                       # how long you want to generate
    Generated_midi_name = 'generated_example/test_RNN_001.midi'              # generated files name


### Note 
#### Other .ipynb files are used for experiments.
