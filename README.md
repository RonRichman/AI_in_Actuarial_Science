# AI_in_Actuarial_Science

Code accompanying the paper "AI in Actuarial Science", available from SSRN here:

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3218082

Folders are provided for each of the following examples:

- Mortality modelling
- Non-life pricing 
- Telematics

The examples in the Non-life pricing folder rely on the Poisson Deviance loss function which is not standard with Keras.

This function has been provided in the file 'poisson_dev.py'. One easy way to incorporate this into Keras is to add the code within this file to the Keras installation directly. This can be done by searching for the file 'losses.py' and copying the code into the file.

Model weights have been provided in each folder. Running the scripts should reproduce similar models, but the results mights differ due to some random variation during the fitting process, such as dropout or the random initialization of the network weights.

