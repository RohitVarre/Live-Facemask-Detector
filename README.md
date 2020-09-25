# Mask-Detector
Motivation: 

The COVID-19 pandemic has reshaped life as we know it. Many of us are staying home, avoiding people on the street and changing daily habits, like going to school or work, in ways we never imagined. While we are changing old behaviours, there are new routines we need to adopt. First and foremost is the habit of wearing a mask or face covering whenever we are in a public space.
  1. Masks and face coverings can prevent the wearer from transmitting the COVID-19 virus to others and may provide some protection to the wearer. Multiple studies have shown        that face coverings can contain droplets expelled from the wearer, which are responsible for the majority of transmission of the virus.
  2. Many people with COVID-19 are unaware they are carrying the virus. It is estimated that 40% of persons with COVID-19 are asymptomatic but potentially able to transmit the      virus to others..
  3. Disease modeling suggests masks worn by significant portions of the population, coupled with other measures, could result in substantial reductions in case numbers and          deaths.
  
Therefore, there is a need to make wearing a mask as compulsion to prevent the spread of the virus especially in public places like shopping malls.

Data:

Data augmentation techniques are used to build a masked dataset. Images of masked people from the internet can also be added to diversify the dataset. It is very important that the data consists of primarily the front face as we are using the frontal face cascade. The model can be extended to various face angles (not just frontal face) by feeding the relevant data and using relevant cascades. In this project frontal face is focussed as it has major applications.

Model:

This model is made by transfer learning from the famous MobileNet model. OpenCV is used to capture live input from the webcam. HAAR cascade is used to detect the frontal face. 
