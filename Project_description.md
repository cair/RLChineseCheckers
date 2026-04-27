## Additional Info
The README.md file provided in this repo contains game rules for the base chinese checkers implementation. The final project for this course is to implement an RL agent to participate in a student tournament among all students taking the course. 

## Grading
Grading is broken down as follows:
* Tournament placement: 50%
* Running agent that can play a human: 30%
* 5 Page report: 20%

In addition to the information provided in the README, i sent an email to the teacher with further questions:

## FAQ response:
I have a list of questions regarding the chinese checkers project. If these are too many questions to answer via email, I can swing by your office when you have time.

Algorithm:
Can we use stablebaseline implementations of RL algorithms for the agents, or do we have to make the agents from scratch in torch?

Teachers Answer: No restrictions. However, a from -scratch implementation will get higher points in the report section. 


Tournament Format:
What is the tournament format like? Knockout, Roundrobin or other?

Teachers Answer: I have planned for Round Robin. 

Game rules:
What is the board size and the number of players per game?

Teachers Answer: 
Board size is constant as given. Number of opponents can be minimum of 1 and maximum of 5.

Participating without a team:
I am making an individual submission. How many agents must I submit to each game? If multiple agents should be submitted, can I submit duplicates of the same agent?

Teachers Answer:
Only one agent per team. Same if you are a team of size 1.


Report:
What should the report include? Should background on reinforcement learning and SotA be included in the 5-page report?

Teachers answer:
I will put up a list of points you need to cover in the report soon. Primarily: chosen algorithm(s), reward design, training mechanism, numbers (average time per move, average move per game, average reward etc.) and finally your score as obtained in the tournament. 


## Further information on execution and wishes for the project.
* I have a limited time to execute this project as it is a class I take in addition to my master thesis with under a month to delivery. 

* I want to do some variation of selfplay  for training. Probably with a cirriculum. In my mind I had the idea of running 1 stage of empty board to learn movement and self hopping and then move to opponent play.

* I want to maximise grade potential while minimizing execution and training time. 

* The implementation should be cleanly modularized into and environment.py file, train.py, evaluate.py and then the trained checkpoint must be compatible with the existing scripts. Do not touch the existing code files for this project.

* Appart from this I would like an agent that is able to play well, and learns to setup jump chains to reach the goal fast. It must also be able to play a different range of opponents, so this should be incorporated in training to be robust against a variety of number of opponents.
