# **Pacman Multi-Agent Systems**
This project introduces **multi-agent decision-making** techniques to model interactions between Pacman and ghosts, balancing strategy and survival.

## **License**
This project is for educational purposes and follows the **Berkeley AI Pacman Project** framework. <br/>
Please note that the project has been solved in teams of 2:
- Melisa Marian's work is marked under `@Author: Melisa Marian`
- Iulia Ana Anca's work is marked under `#Iulia Anca`

## **Overview**
This project extends basic search strategies to multi-agent environments, applying **game theory** techniques like **Minimax, Alpha-Beta Pruning, and Expectimax** to optimize Pacman's gameplay.

### **Implemented Algorithms**
- **Reflex Agent:** A basic agent that reacts to food and ghosts
- **Minimax Search:** Computes an optimal strategy against adversarial ghosts
- **Alpha-Beta Pruning:** Optimizes Minimax by pruning unnecessary branches
- **Expectimax Search:** Handles probabilistic ghost behavior
- **Improved Evaluation Function:** Enhances Pacman’s decision-making

## **How to Run the Multi-Agent Agents**
Test different strategies by running:

- **Reflex Agent:**
  ```bash
  python pacman.py -p ReflexAgent -l mediumClassic -k 1
  ```
- **Minimax Agent:**
  ```bash
  python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
  ```
- **Alpha-Beta Pruning Agent:**
  ```bash
  python pacman.py -p AlphaBetaAgent -l minimaxClassic -a depth=4
  ```
- **Expectimax Agent:**
  ```bash
  python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
  ```
- **Improved Evaluation Agent:**
  ```bash
  python pacman.py -p BetterEvaluationAgent -l smallClassic
  ```

- Use `-h` for a list of available options:
  ```bash
  python pacman.py -h
  ```
  
- Run the autograder to test the given implementation:
```bash
python autograder.py
```

## **File Structure**
- **`multiAgents.py`** – Implements multi-agent decision-making algorithms
- **`pacman.py`** – Main game engine
- **`util.py`** – Helper functions for data structures
