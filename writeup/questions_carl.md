# Questions for Carl (Meeting 04/02/2021)

- Can we use the current train / val / test split of solar cycles (21, 23),
  22 and 24, respectively?
  - In Carl's SVM, he split by alternating years instead of solar cycles
    since different cycles have different magnitudes
    Could include solar cycle phase as a predictor?
    CMEs happen towards declining phase of solar cycle, maybe we want
    better accuracy here?
- Predicting 1 point vs predicting 24?
  - Ideal is a week-long forecast for power grid! Current state-of-the-art
    is 1-2 days in space weather. CMEs are unpredictable, we just observe
    them and then we predict the 1-5 day travel time. We would likely be
    able to predict a day's worth of info!
    What about other solar wind structures? Can probably see at solar
    minimum.
- Is our Analogue Ensemble written effectively? One (clear) improvement is
  to parallelise the code.
  We can weight other parameters in like the wind speed and other important
  things instead of just using magnetic field strength.
  Run it for every point
- How can we effectively compare the LSTM to the Analogue Ensemble?
  Add a false-positive, etc, and turn LSTM -> clf. Look at percentiles for
  threshold values: 99.9, 99, 97.5!
- What is your opinion on the similarity of solar cycles for prediction?
  Can we take even fewer points as a training set to still have effective
  predictions?
- Do you think our network architecture makes sense for the problem at
  hand?
- What physics are we missing / neglecting from the model?
  Should be good!

  In terms of outputs: B_z is really relevant! Carl will send an equation
  for how much the solar wind affects the Earth.

  spacepy has coordinate transform stuff!
