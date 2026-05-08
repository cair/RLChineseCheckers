#add moves per game, distance per game, time per game, pins in goal per game, win bonus per game
import os
import time
import pandas as pd

def parse_log_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return lines

def extract_game_info(lines):
    game_id = None
    players = {}
    scores = {}

    for line in lines:
        line = line.strip()
        if "GAME CREATED" in line:
            game_id = line.split(']')[1].strip()
        elif "PLAYER JOINED" in line:
            parts = line.split('PLAYER JOINED:')[1].strip().split(' as ')
            player_name = parts[0].strip()
            player_color = parts[1].strip()
            players[player_color] = player_name
        elif "SCORE" in line:
            parts = line.split('SCORE')[1].strip().split(':')
            player_info = parts[0].strip()
            score_info = parts[1].strip()
            player_color = player_info.split(' ')[-1].strip('()')
            score_parts = score_info.split(', ') 
            final_score = float(score_parts[0].split('=')[1])
            time_score = float(score_parts[1].split('=')[1])
            move_score = float(score_parts[2].split('=')[1])
            pin_goal_score = float(score_parts[3].split('=')[1])
            distance_score = float(score_parts[4].split('=')[1])
            win_bonus = float(score_parts[5].split('=')[1]) if len(score_parts) > 5 else 0.0
            scores[player_color] = {
                "final_score": final_score,
                "time_score": time_score,
                "move_score": move_score,
                "pin_goal_score": pin_goal_score,
                "distance_score": distance_score,
                "win_bonus": win_bonus
            }
    return game_id, players, scores

def update_leaderboard(log_folder, leaderboard_file):
    leaderboard = []
    processed_files = set()

    while True:
        log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]
        new_files = [f for f in log_files if f not in processed_files]

        for file_name in new_files:
            file_path = os.path.join(log_folder, file_name)
            game_id, players, scores = extract_game_info(parse_log_file(file_path))
            for color, player_name in players.items():
                score = scores[color]
                existing_entry = next((entry for entry in leaderboard if entry['Player Name'] == player_name), None)
                if existing_entry:
                    existing_entry['Score'] += score.get('final_score', 0)
                    existing_entry['Games Played'] += 1
                    existing_entry['Total Time'] += score.get('time_score', 0)
                    existing_entry['Total Moves'] += score.get('move_score', 0)
                    existing_entry['Total Pins in Goal'] += score.get('pin_goal_score', 0)
                    existing_entry['Total Distance'] += score.get('distance_score', 0)
                else:
                    leaderboard.append({'Player Name': player_name, 'Score': score.get('final_score', 0), 'Total Time': score.get('time_score', 0), 'Total Moves': score.get('move_score', 0), 'Total Pins in Goal': score.get('pin_goal_score', 0), 'Total Distance': score.get('distance_score', 0), 'Games Played': 1})
            processed_files.add(file_name)
        for entry in leaderboard:
            entry['Average Score'] = entry['Score'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Time'] = entry['Total Time'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Moves'] = entry['Total Moves'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Pins in Goal'] = entry['Total Pins in Goal'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
            entry['Average Distance'] = entry['Total Distance'] / entry['Games Played'] if entry['Games Played'] > 0 else 0
        leaderboard.sort(key=lambda x: (-x['Average Score'], x['Games Played'], -x['Average Pins in Goal'], -x['Average Distance']))
        df = pd.DataFrame(leaderboard, columns=['Player Name', 'Score', 'Games Played', 'Average Score', 'Total Time', 'Average Time', 'Total Moves', 'Average Moves', 'Total Pins in Goal', 'Average Pins in Goal', 'Total Distance', 'Average Distance', 'Win Bonus'])
        df.to_csv(leaderboard_file, sep='\t', index=False) 

        print(df.head(10))
        time.sleep(10)  # Check for new log files every 10 seconds

if __name__ == "__main__":
    log_folder = 'games'
    leaderboard_file = 'leaderboard.tsv'
    update_leaderboard(log_folder, leaderboard_file)
    