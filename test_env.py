import sys
import os

# Add the src directory to the Python path so we can import the project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import asyncio
from agents.random_agent import RandomAgent
from rooms.room import Room

# Create a local room with a small number of matches for testing
room = Room(
    run_remote_room=False,
    room_name="Test_Room",
    max_matches=1,
    output_folder="test_outputs",
    save_game_dataset=False,
    save_logs_game=True,
    save_logs_room=True,
)

print("[OK] Room created successfully!")
print(f"   Room name: {room.room_name}")

# Create 4 random agents (Chef's Hat requires exactly 4 players)
agents = [RandomAgent(name=f"RandomAgent_{i}", log_directory=room.room_dir) for i in range(4)]

for agent in agents:
    room.connect_player(agent)
    print(f"   Connected agent: {agent.name}")

print("\n[GAME] Starting game with 1 match...")

# Run the game
asyncio.run(room.run())

print("\n[OK] Game completed successfully!")

