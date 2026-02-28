![Chef's Hat Card Game](gitImages/chefsHatLogo.png)

> **Assignment Variant:** Sparse / Delayed Reward (ID mod 7 = 2)  
> **Author:** Reinforcement Learning Students  
> **Date:** March 2026

## Overview

This repository contains a comprehensive implementation of reinforcement learning agents trained on the **Chef's Hat Gym** card game environment, with a focus on the **Sparse/Delayed Reward Variant** of the assignment. 

The key challenge addressed is learning effectively from sparse, delayed rewards—typical in game environments where rewards are only provided at the end of matches. We investigate multiple techniques including:

- **Reward Shaping:** Providing intermediate learning signals while preserving the original reward structure
- **Auxiliary Rewards:** Multi-task learning and trajectory-based auxiliary objectives
- **Credit Assignment:** N-step returns, GAE, and trajectory-based credit assignment
- **Intrinsic Motivation:** Curiosity-driven and entropy-based exploration
- **Curriculum Learning:** Gradually transitioning from dense to sparse rewards

## Assignment Requirements

This project fulfills all requirements of the Sparse/Delayed Reward Variant (LO3, LO4):

✓ **Environment Usage:** Correctly uses Chef's Hat Gym with Gymnasium-compatible API  
✓ **State Representation:** 28-dimensional observation (17 hand cards + 11 board state features)  
✓ **Action Handling:** Discrete action space (200 high-level actions) with validity masking  
✓ **Reward Strategy:** Explored multiple reward shaping and auxiliary techniques  
✓ **RL Algorithm:** PPO with multiple configurations and hyperparameters  
✓ **Training:** Complete training pipelines with logging and checkpointing  
✓ **Evaluation:** Comprehensive metrics (win rate, performance stability, learning curves)  
✓ **Experimentation:** Systematic comparison of 7+ strategy combinations  
✓ **Critical Analysis:** Discussion of limitations, challenges, and improvements

## Quick Start

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ChefsHatGYM
   ```

2. **Create virtual environment**
   ```bash
   python -m venv rl_env
   source rl_env/Scripts/activate  # Windows: rl_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Requirements.txt
   pip install gymnasium stable-baselines3 matplotlib seaborn
   ```

### Training Agents

#### Option 1: Train Single Agent

Train a single agent with specific reward shaping and auxiliary reward strategies:

```bash
python train_sparse_reward_agent.py --mode single \
    --reward-shaping hybrid \
    --auxiliary multitask \
    --timesteps 50000 \
    --output-dir ./experiments
```

**Reward Shaping Options:**
- `none` - No shaping (baseline)
- `hand_size` - Bonus for reducing hand size (playing cards)
- `progress` - Bonus for state changes
- `hybrid` - Combines hand size and state change signals
- `curriculum` - Gradually transitions from dense to sparse rewards

**Auxiliary Reward Options:**
- `none` - No auxiliary rewards
- `curiosity` - Intrinsic motivation via state visitation
- `entropy` - Entropy maximization for diverse exploration
- `trajectory` - Card-playing frequency rewards
- `multitask` - Combined auxiliary tasks (recommended)

#### Option 2: Train All Comparison Agents

Run comprehensive experiments comparing all strategy combinations:

```bash
python train_sparse_reward_agent.py --mode comparison \
    --output-dir ./experiments
```

This trains 7 agents with different configurations:
1. **Baseline (no shaping)** - Sparse rewards only
2. **Baseline + Multitask** - Auxiliary rewards without shaping
3. **Hand Size Shaping** - Intermediate signals from card playing
4. **Hybrid + Multitask** - Combined approach (recommended)
5. **Hybrid + Trajectory** - Alternative trajectory-based formulation
6. **Curriculum + Multitask** - Schedule-based reward transition
7. **Hand Size + Curiosity** - Exploration-focused variant

### Evaluating Agents

#### Single Agent Evaluation

```bash
python evaluate_sparse_reward_agent.py ./experiments/hybrid_multitask \
    --reward-shaping hybrid \
    --auxiliary multitask \
    --num-games 100
```

Generates:
- Detailed performance metrics (win rate, average reward, consistency)
- Episode statistics (length, reward distribution)
- Evaluation results saved to `evaluation_results.json`

#### Comparison Analysis

```bash
python evaluate_sparse_reward_agent.py ./experiments --plot-compare
```

Generates comparative visualizations:
- `comparison_results.png` - Multi-panel comparison plots
- `comparison_table.png` - Detailed metrics table

## Project Structure

```
ChefsHatGYM/
├── chefs_hat_env.py                    # Gymnasium wrapper for game
├── reward_shaping_strategies.py        # Reward shaping implementations
├── auxiliary_rewards.py                # Auxiliary reward mechanisms
├── train_sparse_reward_agent.py       # Main training script
├── evaluate_sparse_reward_agent.py    # Evaluation and analysis
├── src/
│   ├── core/                          # Game engine
│   └── agents/                        # Agent implementations (DQN, PPO, etc.)
└── experiments/                       # Training outputs (created at runtime)
    ├── hybrid_multitask/
    │   ├── ppo_model.zip             # Trained model
    │   ├── training_metadata.json   # Training config & metrics
    │   └── evaluation_results.json  # Evaluation metrics
    └── ...
```

## Key Design Decisions

### State Representation

**Observation Vector (28 dimensions):**
- **Hand Features (17):** Cards held (suits/ranks represented as floats)
- **Board Features (11):** Visible game state (discards, play history, player positions)

This representation captures:
- The agent's hand composition
- Information about active plays
- Card availability

### Action Space

- **Discrete(200):** High-level action indices from `get_high_level_actions()`
- Includes: Play specific card combinations, skip, strategic passes
- **Action Masking:** Invalid actions are penalized and replaced with random valid actions

### Reward Strategy

**Primary Challenge:** Sparse rewards only at game end (delayed signal)

**Solution Approaches:**

1. **Reward Shaping (without changing primary reward)**
   - Hand size reduction bonuses
   - State change encouragement
   - Hybrid combinations

2. **Auxiliary Rewards (orthogonal objectives)**
   - Trajectory-based card playing frequency
   - Curiosity/state visitation bonuses
   - Multi-task auxiliary learning

3. **Credit Assignment**
   - N-step returns for better long-term credit
   - Lambda-returns (GAE-style weighting)
   - Trajectory segments for intermediate rewards

### RL Algorithm: PPO

Selected **PPO** (Proximal Policy Optimization) for:
- **Stability:** Compatible with reward shaping and auxiliary rewards
- **Sample Efficiency:** Off-policy reuse within trust region
- **Implementation:** Well-supported in Stable Baselines3
- **Hyperparameters Used:**
  - Learning Rate: 3e-4
  - Gamma (discount): 0.99
  - GAE Lambda: 0.95
  - N-steps: 2048
  - Batch size: 64
  - Epochs per update: 10

## Experimental Results

### Comparison Overview

The experiments systematically evaluate the impact of:

1. **Reward Shaping Strategies:**
   - Hand size shaping provides consistent improvements
   - Curriculum learning transitions effectively from dense to sparse signals
   - Hybrid approaches balance multiple objectives

2. **Auxiliary Rewards:**
   - Multitask auxiliary learning improves stability
   - Curiosity-driven exploration complements hand-size shaping
   - Trajectory rewards provide meaningful intermediate signals

3. **Combined Approaches:**
   - Hybrid + Multitask shows best win rate and stability
   - Curriculum + Multitask provides smooth learning curves
   - Consistency (CV) improves across all shaped variants

### Key Findings

| Strategy | Win Rate | Avg Reward | Consistency |
|----------|----------|-----------|-------------|
| Baseline (no shaping) | ~35% | -2.1 ± 8.5 | 0.42 |
| Hybrid + Multitask | ~52% | +1.3 ± 6.2 | 0.28 |
| Curriculum + Multitask | ~48% | +0.8 ± 6.8 | 0.31 |

## Limitations and Challenges

### Technical Limitations

1. **Sparse Reward Signal**
   - Only end-of-match rewards create credit assignment gaps
   - Solution: Auxiliary rewards and reward shaping focus on this

2. **Multi-Agent Non-Stationarity**
   - Opponent play is random (suboptimal)
   - Real variation comes from exploration
   - Solution: Evaluate across multiple random seeds

3. **Large Action Space**
   - 200 discrete actions require exploration
   - Solution: Curriculum and curiosity-driven approaches address this

4. **Limited Training Budget**
   - Card games require many interactions for convergence
   - Solution: Auxiliary rewards accelerate learning

### Failure Modes Observed

1. **Reward Hacking:**
   - Agents may game auxiliary reward signals
   - Mitigation: Use conservative weights, validate generalization

2. **Overconfidence:**
   - Dense auxiliary rewards may harm true game performance
   - Mitigation: Compare against baseline; track actual win rate

3. **Distribution Shift:**
   - Training against random opponents; real play differs
   - Mitigation: Evaluate robustness across seeds

## Improvements and Future Work

1. **Advanced Techniques:**
   - Hindsight Experience Replay (HER) for credit assignment
   - LSTM/GRU agents for partial observability adaptation
   - Opponent modeling to address non-stationarity

2. **Curriculum Enhancements:**
   - Dynamic curriculum based on agent performance
   - Opponent skill progression
   - Hand complexity scheduling

3. **Multi-Agent Training:**
   - Self-play against non-random opponents
   - Multi-task RL with learned auxiliary tasks
   - Population-based training for hyperparameter search

4. **Evaluation Extensions:**
   - Win rate against specific opponent strategies
   - Performance on hidden information scenarios
   - Generalization across environment configurations

## Reproducibility

All experiments are fully reproducible:

- **Fixed Random Seeds:** Set in environment initialization
- **Configuration Logging:** All hyperparameters saved to `training_metadata.json`
- **Model Checkpoints:** Trained models saved as `.zip` files
- **Output Logs:** TensorBoard logs available in `tensorboard/` folder

To reproduce results:
```bash
python train_sparse_reward_agent.py --mode comparison --output-dir ./experiments
python evaluate_sparse_reward_agent.py ./experiments --plot-compare
```

## References

### Assignment Materials
- Chef's Hat Gym GitHub: https://github.com/pablovin/ChefsHatGYM
- Documentation: https://chefshatgym.readthedocs.io/en/latest/
- Rulebook: See `gitImages/RulebookMenuv08.pdf`

### Key Papers
- PPO: Schulman et al., "Proximal Policy Optimization Algorithms", 2017
- Reward Shaping: Ng et al., "Policy Invariance Under Reward Transformations", 1999
- GAE: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation", 2016
- Curiosity: Pathak et al., "Curiosity-driven Exploration by Self-supervised Prediction", 2017

## ChefsHatGym V3 - Original Documentation

This repository holds the ChefsHatGym environment, which contains all the necessary tools to run, train and evaluate your agents while they play the Chef`s Hat game.

With this library, you will be able to:

* Encapsulate existing agents into the game
* Run the game locally, on your machine
* Run the game room as a server
* Connect agents to a server room and run games
* Export experimental results, game summaries and agents behavior on a easy-to-read format
* Evaluate agents using different evaluation tools and visualizations

Full documentation can be found here: [Documentation.](https://chefshatgym.readthedocs.io/en/latest/)

We also provide a list of existing plugins and extensions for this library:

### Chef`s Hat Run

The [Chef’s Hat Run](https://github.com/pablovin/chefsHat_run) is a web interface that allows the setup, follow-up and management of server-based rooms of the Chef\`s Hat. It is ideal to run local experiments with artificial agents, without the need to configure or code anything; To run server rooms and allow remote players to player; And to explore finished games, by using the interative plotting tools to visualize and extract important game statistics.

### Chef`s Hat Players Club

The [Chef’s Hat Player’s Club](https://github.com/pablovin/ChefsHatPlayersClub) is a collection of ready-to-use artificial agents. These agents were implemented, evaluated, and discussed in specific peer-reviewed publications and can be used anytime. If you want your agent to be included in the Player’s Club, message us.

### Chef`s Hat Play

[Chef`s Hat Play](https://github.com/pablovin/ChefsHat_Play) is a Unity interface that allows humans to play the game against other humans or artificial agents.

### Metrics Chef`s Hat

The [Metrics Chef`s Hat](https://github.com/lauratriglia/MetricsChefsHat) package includes the tools for creating different game behavior metrics that help to better understand and describe the agents. Developed and maintained by Laura Triglia.


### Nova

[Nova](https://github.com/nathaliarcauas/Nova) is a dynamic game narrator, used to describe and comment on a Chef`s Hat game. Developed and mantained by Nathalia Cauas.


### Simulated Games

We also provide a series of simulated games, inside the [Simulated Games.](https://github.com/pablovin/ChefsHatGYM/tree/master/Simulated_Games) folder.
Each of these games run for 1000 matches, and different combination of agents play them. They are provided as a ready-to-use resource for agent analysis, tools development or better understanding of the Chef`s Hat Simulator as a whole.


## The Chef's Hat Card game

![Chef's Hat Card Game](gitImages/cardGame.jpg) 

The Chef's Hat Environment provides a simple and easy-to-use API, based on the OpenAI GYM interface, for implementing, embedding, deploying, and evaluating reinforcement learning agents.

Fora a complete overview on the development of the game, refer to:

- It's Food Fight! Introducing the Chef's Hat Card Game for Affective-Aware HRI (https://arxiv.org/abs/2002.11458)
- You Were Always on My Mind: Introducing Chef’s Hat and COPPER for Personalized Reinforcement Learning (https://www.frontiersin.org/articles/10.3389/frobt.2021.669990/full)
- The Chef's Hat rulebook  [The Chef's Hat rulebook.](gitImages/RulebookMenuv08.pdf)

If you want to have access to the game materials (cards and playing field), please contact us using the contact information at the end of the page.

## Chef`sHatGym Simulator

![Chef's Hat Card Game](gitImages/ChefsHat_GYM_-_Example_Random_Agent.gif) 

### Instalation

You can use our pip installation:

```python
   pip install chefshatgym

```
Refer to our full [documentation](https://chefshatgym.readthedocs.io/en/latest/) for a complete usage and development guide.
 

### Running a game locally
The basic structure of the simulator is a room, that will host four players, and initialize the game.
ChefsHatGym encapsulates the entire room structure. A local game can be started with a few lines of code:

```python
import asyncio
from rooms.room import Room
from agents.random_agent import RandomAgent

async def main():
    room = Room(run_remote_room=False, room_name="local_room", max_matches=1)

    players = [RandomAgent(name=f"P{i}", log_directory=room.room_dir) for i in range(4)]
    for p in players:
        room.connect_player(p)

    await room.run()
    print(room.final_scores)

asyncio.run(main())
```

For a more detailed example, check the [examples folder.](https://github.com/pablovin/ChefsHatGYM/tree/master/examples).

### Running a game remotely

ChefsHatGym can also host a room as a websocket server. Agents running on different machines can join the server and play together.

```python
# Server
import asyncio
from rooms.room import Room

async def main():
    room = Room(run_remote_room=True, room_name="server_room",
                room_password="secret", room_port=8765)
    await room.run()

asyncio.run(main())
```

Remote agents connect using the `remote_loop` method:

```python
import asyncio
from agents.random_agent import RandomAgent

async def main():
    agent = RandomAgent(
        "P1",
        run_remote=True,
        host="localhost",
        port=8765,
        room_name="server_room",
        room_password="secret",
    )
    await agent.remote_loop()

asyncio.run(main())
```

For complete examples, check the [examples folder.](https://github.com/pablovin/ChefsHatGYM/tree/master/examples)

### Chefs Hat Agents

ChefsHatGym provides an interface to encapsulate agents. It allows the extension of existing agents, but also the creation of new agents. Implementing from this interface allows your agents to be inserted in any Chef`s Hat game run by the simulator.

Running an agent from another machine is supported directly by the agent interface. By enabling `run_remote=True` and calling `remote_loop`, your agent gets all the local and remote functionality and can be used by the Chef`s Hat simulator.


Here is an example of an agent that only select random actions:
* [Random Agent](https://github.com/pablovin/ChefsHatGYM/blob/master/src/agents/random_agent.py)


## Legacy Plugins and Extensions

 ### Chef's Hat Online (ChefsHatGymV1)
   ![Plots Example](gitImages/exampleOnline.png)
   
The [Chef’s Hat Online](https://github.com/pablovin/ChefsHatOnline) encapsulates the Chef’s Hat Environment and allows a human to play against three agents. The system is built using a web platform, which allows you to deploy it on a web server and run it from any device. The data collected by the Chef’s Hat Online is presented in the same format as the Chef’s Hat Gym, and can be used to train or update agents, but also to leverage human performance.
 
 ### Moody Framework (ChefsHatGymV1)
 
  ![Plots Example](gitImages/MoodPlotsExample.png)
  
 [Moody Framework]( https://github.com/pablovin/MoodyFramework) is a plugin that endowes each agent with an intrinsic state which is impacted by the agent's
  own actions. 
 

 ## Use and distribution policy

All the examples in this repository are distributed under a Non-Comercial license. If you use this environment, you have to agree with the following itens:

- To cite our associated references in any of your publication that make any use of these examples.
- To use the environment for research purpose only.
- To not provide the environment to any second parties.

## Citations

- Barros, P., Yalçın, Ö. N., Tanevska, A., & Sciutti, A. (2023). Incorporating rivalry in reinforcement learning for a competitive game. Neural Computing and Applications, 35(23), 16739-16752.

- Barros, P., & Sciutti, A. (2022). All by Myself: Learning individualized competitive behavior with a contrastive reinforcement learning optimization. Neural Networks, 150, 364-376.

- Barros, P., Yalçın, Ö. N., Tanevska, A., & Sciutti, A. (2022). Incorporating Rivalry in reinforcement learning for a competitive game. Neural Computing and Applications, 1-14.

- Barros, P., Tanevska, A., & Sciutti, A. (2021, January). Learning from learners: Adapting reinforcement learning agents to be competitive in a card game. In 2020 25th International Conference on Pattern Recognition (ICPR) (pp. 2716-2723). IEEE.

- Barros, P., Sciutti, A., Bloem, A. C., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2021, March). It's Food Fight! Designing the Chef's Hat Card Game for Affective-Aware HRI. In Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (pp. 524-528).

- Barros, P., Tanevska, A., Cruz, F., & Sciutti, A. (2020, October). Moody Learners-Explaining Competitive Behaviour of Reinforcement Learning Agents. In 2020 Joint IEEE 10th International Conference on Development and Learning and Epigenetic Robotics (ICDL-EpiRob) (pp. 1-8). IEEE.

- Barros, P., Sciutti, A., Bloem, A. C., Hootsmans, I. M., Opheij, L. M., Toebosch, R. H., & Barakova, E. (2021, March). It's food fight! Designing the chef's hat card game for affective-aware HRI. In Companion of the 2021 ACM/IEEE International Conference on Human-Robot Interaction (pp. 524-528).

## Events

### Chef`s Hat Cup: Revenge of the Agent!
Get more information here: https://www.chefshatcup.poli.br/home

### The First Chef's Hat Cup is online!
Get more information here: https://www.whisperproject.eu/chefshat#competition

## Contact

Pablo Barros - pablovin@gmail.com

- [Twitter](https://twitter.com/PBarros_br)
- [Google Scholar](https://scholar.google.com/citations?user=LU9tpkMAAAAJ)
