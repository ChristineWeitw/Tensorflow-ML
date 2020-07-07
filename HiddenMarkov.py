import tensorflow_probability as tfp
%tensorflow_version 2.x

# PREPARE DATA
tfd = tfp.distributions
## 1. State  [cold,hot]
initial_dist = tfd.Categorical(probs=[0.8, 0.2])
## 2. Transaction  [cold,hot]
transaction_dist = tfd.Categorical(probs=[[0.7,0.3],[0.2,0.8]])
## 3. Observation  (loc->mean;scale->SD)
observation_dist = tfd.Normal(loc=[0.,15.],scale=[5.,10.])

# CREATE MODEL
 ##num_steps = the amount of days you want to predict
HMmodel = tfd.HiddenMarkovModel(
    initial_distribution=initial_dist,
    transition_distribution=transaction_dist,
    observation_distribution=observation_dist,
    num_steps=7)

  ##how to see the values the model predict
mean = HMmodel.mean()
with tf.compat.v1.Session() as sess:
    print(mean.numpy())

