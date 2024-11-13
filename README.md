# Snake AI Bot

This repository contains a Snake AI bot that learns to play the classic Snake game autonomously using reinforcement learning. The bot improves over time by optimizing its strategy to achieve higher scores, avoiding obstacles, and aiming to consume food effectively.

This project is inspired by [Patrick Loeber's YouTube tutorial](https://youtu.be/L8ypSXwyBds?si=7WZQeQWUsDpe3osa), which covers the fundamentals of implementing reinforcement learning in a Snake game.

## Features

- **Reinforcement Learning**: The AI learns over time through trial and error, making it more effective with each game it plays.
- **Autonomous Gameplay**: The bot navigates the Snake game independently, avoiding obstacles and improving its score.
- **Python Implementation**: Built using Python, with `Pygame` for the game interface and `NumPy` for numerical operations.
- **Flexible Code Structure**: Modular design for easy adjustments and experimentation.

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed along with the following libraries:

- `pygame` for the game interface
- `numpy` for data handling

Install the dependencies using pip:
```bash
pip install pygame numpy
```
## Installation
- Clone the Repository
```bash
git clone https://github.com/Weirdnemo/snake-ai.git
cd snake-ai
```
- Run the Bot
```bash
python agent.py
```
The bot will start training and playing the Snake game. Watch it learn and improve with each iteration.

## How it Works
The bot uses a reinforcement learning approach to play Snake. Here’s a breakdown of the process:
- Game State Representation: The bot observes its surroundings (position, food location, and direction) and constructs a game state.
- Action Decision: The AI decides its next move based on the current state and attempts to maximize rewards.
- Reward System: The bot receives positive rewards for eating food and negative rewards for crashing or moving away from the food.
- Learning Process: Over time, the bot uses past experiences to improve decision-making and optimize its actions for better scores.

## Project Structure
- game.py: Contains the core game mechanics and functions to control the Snake game.
- model.py: Defines the neural network model used to predict the best actions.
- utils.py: Utility functions for data processing, reward calculation, and model training.
- agent.py: Main file that initializes and runs the game with the AI.

## Customization
Feel free to experiment with the reward structure, model parameters, and game speed to optimize the bot’s performance or adapt it for different versions of the Snake game.

## Acknowledgments
This project was inspired by the Patrick Loeber's YouTube tutorial, which provides a great introduction to reinforcement learning concepts applied to game AI.

## Contact
For questions or suggestions, please reach out by opening an issue on this repository or contacting me at [nimesh.chn@gmail.com].
