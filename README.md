# Hand Motion Intention Prediction (hmip_ros)

In human-robot collaboration (HRC) settings, hand motion intention prediction (HMIP) plays a pivotal role in ensuring prompt decision-making, safety, and an intuitive collaboration experience. Precise and robust HMIP with low computational resources remains a challenge due to the stochastic nature of hand motion and the diversity of HRC tasks. This proposed framework combines hand trajectories and gaze data to foster robust, real-time HMIP with minimal to no training. A novel 3D vector field method is introduced for hand trajectory representation, leveraging minimum jerk trajectory predictions to discern potential hand motion endpoints. This is statistically combined with gaze fixation data using a weighted Naive Bayes Classifier (NBC). Acknowledging the potential variances in saccadic eye motion due to factors like fatigue or inattentiveness, we incorporate stationary gaze entropy to gauge visual concentration, thereby adjusting the contribution of gaze fixation to the HMIP. Empirical experiments substantiate that the proposed framework robustly predicts intended endpoints of hand motion before at least 50% of the trajectory is completed. It also successfully exploits gaze fixations when the human operator is attentive and mitigates its influence when the operator loses focus. A real-time implementation in a construction HRC scenario (collaborative tiling) showcases the intuitive nature and potential efficiency gains to be leveraged by introducing the proposed HMIP into HRC contexts. A deeper explanation of this work can be found in our draft paper: [Link to Draft Paper](https://drive.google.com/file/d/1ztWVfJ50tQnFpHZj4Nf5Jow-s6y52FKk/view?usp=sharing)

Watch Video:

<div align="center">
      <a href="https://youtu.be/6foeRxCCqRk?si=MTCU8gc60DLDNtvs">
     <img 
      src="https://i.imgur.com/R3xlUQ3.jpg" 
      alt="Video on Draft Paper" 
      style="width:75%;">
      </a>
    </div>


## Package Descriptions


## Instructions and tutorials

1. Testing pipeline with dataset
2. Running pipeline 
3. Running prediction nodes

