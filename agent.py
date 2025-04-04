import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001  # Significantly reduced learning rate for more stable training


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 100  # Much higher starting randomness (pure exploration)
        self.epsilon_min = 0  # Higher minimum randomness
        self.epsilon_decay = 0.998  # Much slower decay
        self.gamma = 0.9  # Discount rate
        self.memory = deque(maxlen=MAX_MEMORY)
        # Simplified network architecture
        self.model = Linear_QNet(11, 128, 64, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        self.last_10_scores = deque(maxlen=10)
        self.evaluate_interval = 100

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game._is_collision(point_r)) or
            (dir_l and game._is_collision(point_l)) or
            (dir_u and game._is_collision(point_u)) or
            (dir_d and game._is_collision(point_d)),

            # Danger right
            (dir_u and game._is_collision(point_r)) or
            (dir_d and game._is_collision(point_l)) or
            (dir_l and game._is_collision(point_u)) or
            (dir_r and game._is_collision(point_d)),

            # Danger left
            (dir_d and game._is_collision(point_r)) or
            (dir_u and game._is_collision(point_l)) or
            (dir_r and game._is_collision(point_u)) or
            (dir_l and game._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # Convert boolean values to integers before storing
        state = np.array(state, dtype=int)
        next_state = np.array(next_state, dtype=int)
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE // 10:  # Don't train until we have some minimum experience
            return

        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Convert boolean values to integers before training
        state = np.array(state, dtype=int)
        next_state = np.array(next_state, dtype=int)
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Apply epsilon decay only after some initial games for pure exploration
        if self.n_games > 50:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        final_move = [0, 0, 0]

        # Force more exploration
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Convert state to tensor properly
            state0 = torch.tensor(state, dtype=torch.float)
            with torch.no_grad():  # No need to track gradients during prediction
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    # Add evaluation mode toggle
    training_mode = True
    checkpoint_interval = 500

    print("Starting training...")

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory only if in training mode
        if training_mode:
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1

            if training_mode:
                agent.train_long_memory()

            if score > record:
                record = score
                if training_mode:
                    agent.model.save('record_model.pth')
                    print(f"New record! {record}")

            print(f'Game {agent.n_games}, Score {score}, Record {record}, Epsilon {round(agent.epsilon, 2)}')
            agent.last_10_scores.append(score)

            if agent.n_games % 10 == 0:
                avg_score = sum(agent.last_10_scores) / len(agent.last_10_scores)
                print(f"Last 10 games average score: {avg_score:.2f}")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            # Periodically save checkpoint
            if agent.n_games % checkpoint_interval == 0 and training_mode:
                agent.model.save(f'model_checkpoint_{agent.n_games}.pth')

                # Toggle evaluation mode periodically to see how the model performs
                if agent.n_games % agent.evaluate_interval == 0:
                    old_epsilon = agent.epsilon
                    agent.epsilon = 0  # Turn off randomness to see actual performance
                    print(f"Evaluation mode for 10 games...")
                    evaluation_scores = []
                    for _ in range(10):
                        # Run one full game in evaluation mode
                        eval_game = SnakeGameAI()
                        eval_done = False
                        while not eval_done:
                            state = agent.get_state(eval_game)
                            move = agent.get_action(state)
                            _, eval_done, eval_score = eval_game.play_step(move)
                        evaluation_scores.append(eval_score)
                        print(f"Evaluation game score: {eval_score}")

                    print(f"Evaluation average: {sum(evaluation_scores) / 10:.2f}")
                    agent.epsilon = old_epsilon  # Restore epsilon


if __name__ == '__main__':
    train()