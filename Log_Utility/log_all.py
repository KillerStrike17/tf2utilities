from .libraries import *

#@params
#   file_path : Location to store the log file
#   project_name : Name of the Log file
#   Model_update : A model Update bit, 0 representing No update in model
#                : and 1 representing model is being updated.

def logmodel(file_path, project_name, learning_rate, epochs, batchsize, accuracy, loss, model_summary, hyperparameters,Augmentation,Remarks,Model_update):
	file = file_path + "/" + project_name
	with open("{}.txt".format(file), "a") as myfile:
		now = datetime.now()
		timestamp = datetime.timestamp(now)
		print(now)
		print("timestamp =", timestamp)
		myfile.write("\n\n")
		myfile.write("Date and Time " + str(now) + "\n")

		if(model_update == 'y' or model_update == 'yes' or model_update == 'Y'):
			myfile.write("Model Summary \n")
			myfile.write(str(model_summary) + "\n")

		myfile.write("Learning_rate :- ")
		myfile.write(str(learning_rate) + "\n")
		myfile.write("Epochs :- ")
		myfile.write(str(epochs) + "\n")
		myfile.write("BatchSize :- ")
		myfile.write(str(batchsize) + "\n")
		myfile.write("Accuracy :- ")
		myfile.write(str(accuracy) + "\n")
		myfile.write("loss :- ")
		myfile.write(str(loss) + "\n")
		myfile.write("hyperparameters :- ")
		myfile.write(str(hyperparameters) + "\n")
		myfile.write("Augmentation :- ")
		myfile.write(str(Augmentation) + "\n")
		myfile.write("Remarks :- ")
		myfile.write(str(Remarks) + "\n")
