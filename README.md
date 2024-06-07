<div align="center">
<img src="https://hermes.digitalinnovation.one/assets/diome/logo-full.svg" alt="Logo Bootcamp" width="80">
<h1> Coding The Future Vivo <br> Python AI Backend Developer </h1>
<img src="https://hermes.dio.me/files/assets/ef695d25-f647-45eb-b1ad-a25c124b28ca.png" alt="Logo Bootcamp" width="220">
</div>

# natty_or_not
Final challenge of the DIO Python Backend bootcamp. The objective is to showcase the myriad AI functionalities by using whicever AI tools currently available to produce an end-product.
The sky seems to be the limit here, so I will begin this project by feeding this exact information to an AI platform, namely Blackbox AI. This is the initial prompt:

'''
Hello Blackbox. I am tasked with solving a challenge for a data science bootcamp. In this challenge, one must use whatever free AI tools currently available to develop an "end product". For example: provide a prompt about a subject so that an AI tool can expand on that subject in the form of a script for a podcast. Then, with that script, use an AI tool to generate an AI-voiceover of the script, emulating a real podcaster talking about said subject. This is just an example. In my case, I would like to use the following components:

- an AI tool for synthetic data generation such as gretel, mdclone or hazy. I want to generate a timeseries of sea surface temperature measurements on a point in the ocean, emulating data that could come from a meteoceanographic buoy with sensors installed. In particular, I would like a dataset of 1 year of daily SST measuments at a fictional point near the sea of Rio de Janeiro, Brazil.
- an AI tool for code generation and documentation such as replit, codiga or code9, which should generate a C++ code that will ingest the data create in the previous section and manipulate it by calculating basic statistics of the time series such as mean and standard deviation, which will then be used to define an interval of mean +/- 2 times the standard deviation. The dataset should then be filtered by removing outliers i.e. removing data points that are outside this interval.
- an AI tool for summarization and insights such as fathom, xembly or hume, which will then receive the original dataset and the new, filtered dataset, an generate a textual conclusion stating how many points were removed by this filtering process.
- finally, an AI tool for text-to-speech functionality that will create a voice-over of the synthesis created by the previous AI tool.

Do you think this is feasible? And, if so, can you help me achieve it?
'''

Turns out that the Blackbox AI _could_ indeed help me achieve this:

'''
What an exciting project! I'm happy to help you explore the feasibility of this challenge and provide guidance on how to achieve it using various AI tools.

Feasibility: After reviewing your requirements, I believe it's entirely feasible to complete this project using the mentioned AI tools. Each component you've outlined can be achieved using the corresponding AI tools, and the workflow seems well-structured.

Component 1: Synthetic Data Generation You can use Gretel, MDClone, or Hazy to generate a timeseries of sea surface temperature measurements. Let's assume you choose Gretel. You can create a synthetic dataset with the following parameters:

Dataset type: Time series
Data points: 365 (1 year of daily measurements)
Variables: Sea surface temperature (SST)
Location: Fictional point near the Sea of Rio de Janeiro, Brazil
Distribution: You can choose a suitable distribution for SST, such as a normal distribution with a mean and standard deviation based on real-world data.
Component 2: Code Generation and Documentation You can use Replit, Codiga, or Code9 to generate a C++ code that ingests the synthetic data and performs the required calculations. Let's assume you choose Replit. You can create a new Replit project, specify the input dataset, and ask the AI to generate a C++ code that:

Reads the input dataset
Calculates the mean and standard deviation of the SST time series
Defines an interval of mean +/- 2 times the standard deviation
Filters the dataset by removing outliers (data points outside the interval)
Component 3: Summarization and Insights You can use Fathom, Xembly, or Hume to generate a textual conclusion based on the original and filtered datasets. Let's assume you choose Fathom. You can upload both datasets to Fathom and ask the AI to generate a summary that includes:

The number of data points removed by the filtering process
A brief description of the filtering process
Any other relevant insights or statistics about the datasets
Component 4: Text-to-Speech You can use various text-to-speech AI tools, such as Amazon Polly, Google Text-to-Speech, or IBM Watson Text to Speech, to generate a voice-over of the summary created by Fathom.

Next Steps: To get started, I recommend the following:

Create a Gretel account and generate the synthetic dataset according to your specifications.
Use Replit to generate the C++ code that ingests the synthetic data and performs the required calculations.
Upload both datasets (original and filtered) to Fathom and generate a summary.
Use a text-to-speech AI tool to create a voice-over of the summary.
If you need help with any of these steps or require more guidance on using these AI tools, feel free to ask, and I'll do my best to assist you.
'''

Might as well follow his suggestions. We hence begin...

#.. STEP 1 - Create a synthetic dataset of SST with Gretel