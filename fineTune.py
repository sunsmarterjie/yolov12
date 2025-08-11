from ultralytics import YOLO

# Load the exported model.
model = YOLO("out/my_experiment/exported_models/exported_last.pt")

model.train(data='data/data.yaml', epochs=5) # This is where the BEANS labeled DATASET GOES

# The results object was causing an AttributeError when trying to access curves_results.
# The relevant metrics are already printed during the training process.
print("Finetuning complete. Check the output above for metrics like mAP50 and mAP50-95.")



#FINETUNING > this is about finding other pretrained datasets with labels already to match your current unlabeled dataset.

#or i can just use lightlytrain for my raw unlabaled images to finetune the pretrained model.