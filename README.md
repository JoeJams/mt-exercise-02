# MT Exercise 2: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/marcamsler1/mt-exercise-02
    cd mt-exercise-02

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

# Reported Changes and Instructions 
<b>Adjusted from above for changes and additions by me </b>

## Steps
    git clone https://github.com/JoeJams/mt-exercise-02.git
    cd mt-exercise-02
> Clone into the forked repository and move to correct the cloned folder

Run to create a new virtual environment (do not forget to activate afterwards):

    ./scripts/make_virtualenv.sh
> I made some minor changes for compatibility with my device (I used git bash but there were still some issues with my device and I had to change some stuff away from UNIX) (I hope it still works for other devices but I couldn't test it, otherwise the original script should be used instead).

Download and install required software:

    ./scripts/install_packages.sh
> No changes here.

Additionally run: 
`pip install -r requirements.txt`
> To make sure all the needed dependencies are downloaded (especially for the python programm later)

Download data: [wow.tgz](https://huggingface.co/datasets/RUCAIBox/Open-Dialogue/resolve/main/wow.tgz)
> I did not manage an automatic download because of some huggingface and xet issues which made manual download a lot easier in this case. After downloading, the folder just needs to be put in the current workdirectory and then the script can be run.

Run to unpack and preprocess data:
    
    ./scripts/download_data.sh
> The tgz file is unpacked and then the target file is moved to the correct folder, and the unneeded files are deleted again. I used the already existing preprocessing files with some minor adjustments so some HTML elements (such as quotation marks) were correctly decoded (I also added a flag into data.py to ignore some of the encoding issues and not raise an error every time and it didn't pose any problems further down the line). Then the vocab limit is applied and the text (total of 10'000 lines) is split into training, validation and test (80-10-10). 

DO NOT RUN (only informative):

    main.py 
>Added a flag that saves perplexities to a csv. The function stored the perplexties in dictionaries while the training is run (as they are output anyways already so I can also store them at the same time) using the epoch as the key and the perplexity as the value. At the end, all these perplexties are saved to a csv file (with the dropout rate in file name and title for easier distinction).

Train the model:

    ./scripts/train.sh
> I only added the flag '--save_perp_log' and then changed some of the values according to the need (i.e. change of dropout, modelname etc.)
(Ultimately I trained with dropout rates of 0.0, 0.2, 0.5, 0.8, 1.0)

Run to create tables and plots for training, validation and test perplexity (note: logs created during training are needed!) 

`python line_chart_ppl.py`

With flags if needed:

        options:
        -h, --help            show this help message and exit
        --log_dir LOG_DIR     Directory where the log files are    
        located (default='tools/pytorch-examplesword_language_model')
        --save_tables         Flag to save the tables as csv files (e.g. as training_perplextiy.csv)
        --save_plots          Flag to save the plots as png files (e.g. as training_perplexity.png)
        --show_plots          Flag to display the plots after creation
        --output_dir          Directory to save the output tables and plots (default='.' (i.e. current directory))

e.g. `python line_chart_ppl.py --save_tables --output_dir target_folder\tables`

> This is the function that can transform the csv files created in the prior step into csv files as wanted in the exercise (i.e. one training, one validation and one test each with the different dropout rates). It can also create the plots for training and validation perplexities. I added some flags for easier handling (mainly for myself). (Note that these functions are not tested for any other data and are therefore not very flexible for anything other than these exact files). 

