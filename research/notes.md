
Dissertation title: “How reliably can uncertainty estimates reflect changes in object detection performance under different environmental conditions?”

Object detection -> use yolo https://docs.ultralytics.com/tasks/detect
We can create our own model, which uses a huge subset of images, or we can use a pretrained model using either yaml or pt. 

We could decide that we want to detect one singular object with importance:
-> vehicles
-> suspicious behaviour
-> weapons* 

The issue with current object detection is that the validity of the predictor can be altered based on key factors like increased object density, reduced lighting, blur.

The idea of detecting weapons could be good -> usually hidden with obscurity and in suboptimal lighting conditions.

Training the model - Annotate objects: Use a tool like Roboflow, Label Studio, or AnyLabeling to draw bounding boxes around your objects.

Obtain around 500 items -> could use ai for this to generate these bounding boxes for the items to save time.
https://www.kaggle.com/datasets/zinkzsa/weapon-and-knife-detection-137k-images
Kaggle dataset, shows bounding boxes for class 0 and 1 weapons with 137,000 images.

we can train the data and use the bounding boxes using a webcam feed that will be live -> using OpenCV -

Steps: 
1) capture the frame from the webcam using OpenCV
2) detect item using YOLO and return a confidence score 0-1
3) estimate the uncertainty:       
    run the model 3-5 times with different orientations and scales of tracked items (weapons) average the score
4) display a bounding box around the item with a colour corresponding to confidence e.g. 1 = Green, 0 = Red

Testing:
1) Degrade the images: 
    blur
    darken
    noiser
    occluded
2) run part 1(2,3) on each image
3) mark every detection as True or False -> create a true box and if our detection overlaps a true box it is classified as a correct prediction.
4) compare uncertainty against correctness:
    measure correlation
    plot graphs
    test multiple methods, changing variables to see if we can achieve a better accuracy (research other methods we can try and implement, instead of relying on just a YOLO prediction)
    

Summarised Steps:
1. Capture: Grab a frame from the webcam (OpenCV)
2. Detect and track: YOLO returns box, class and confidence, plus a track ID that stays the same for the same object across frames:
    (Ultralytics (model.track(..., persist=True))) -- this is the code to give it an ID
3. Single-frame uncertainty: Run YOLO on 3–5 altered copies of the frame and measure how much they disagree
4. Update belief: For each track ID: fade the old belief slightly, combine it with this frame's evidence using Bayes' rule, and store it for next frame
5. Display:	Colour each box by the belief's uncertainty, from green (confident) to red (unsure)	OpenCV

epistemic uncertainty:
use these types of models, -> epistemic object detection -- Fabio at brookes-> good reference for epistemic uncertainty (https://link.springer.com/book/10.1007/978-3-030-63153-6)

we dont have much to understand "liklihood of outputs" rather than classifying inputs: so doing back to front.

uncertainty --> sunflares can be detected even though they arent an object, this is another thing to test and try and mitigate the issues of sunglares in images. 


first step: make object detector now, then further change it to an epistemic object detector

what datasets and objects do we want to detect. 

get several different datasets so we can see distributional shifts.

get at least 2 datasets to use for our object detection.

-- agenda 
Check what models we can use apart from the generic yolo -> do we need multiple models.
How can we update the model in realtime to better the accuracy
- yet to impliment the weapon detection but have gotten 4 datasets from different websites to avoid bias
How can we show the uncertainty when we are given a score is it just 1 minus the score? or are there any other factors we need to check
what do i need to write in the background review for this proposal and project management 



