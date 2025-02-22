import p3.state
import copy
import os

class Stats:
    def __init__(self):
        self.total_frames = 0
        self.skipped_frames = 0
        self.thinking_time = 0
        self.stocks_taken = 0
        self.stocks_lost = 0
        self.damage_recieved = 0
        self.damage_done = 0
        self.prevState = None
        self.game_stocks_lost = 0
        self.game_stocks_taken = 0
        self.games_won = 0
        self.games_lost = 0
        self.states = {}

    def __str__(self):
        if not self.total_frames:
            return ''
        frac_thinking = self.thinking_time * 1000 / self.total_frames
        total_games = self.games_won + self.games_lost
        
        action_states = []
        for i in range(5):
            if len(self.states) >= (i+1):
                key = max(self.states, key=lambda key: self.states[key])
                action_states.append(key)
                self.states.pop(key)

        return '\n'.join(
            ['Total Frames: {}'.format(self.total_frames),
             'Skipped: {} | {}%'.format(self.skipped_frames, round(self.skipped_frames/self.total_frames*100, 2)),
             'Average Thinking Time (ms): {:.6f}'.format(frac_thinking), 
             'Stocks Taken: {} | {:.2f}% Dealt'.format(self.stocks_taken, self.damage_done), 'Stocks Lost: {} | {:.2f}% Recieved'.format(self.stocks_lost, self.damage_recieved), 
             'W/L: {}/{} | {:.2f}%'.format(self.games_won, self.games_lost, (self.games_won/total_games if total_games else 0)),
             'Avg. Stocks Taken Per Game: {}'.format(self.get_average_stocks_taken_game()),
             'Avg. Stocks Lost Per Game: {}'.format(self.get_average_stocks_lost_game()),
             'Top 5 Most Popular Action States',
             '--------------------------------',
             '{} | {} | {} | {} | {}'.format(p3.state.ActionState(action_states[0]), p3.state.ActionState(action_states[1]), p3.state.ActionState(action_states[2]), p3.state.ActionState(action_states[3]), p3.state.ActionState(action_states[4]))])

    def save_readable_results(self, model):
        prefix = '--------------------------------------------------------\n' + model + '\n--------------------------------------------------------\n'
        suffix = '\n\n\n'
        result = str(self)

        filemode = 'w'
        if os.path.exists('./results.txt'):
            filemode = 'a'
        
        print(filemode)
        with open('results.txt', filemode) as f:
            f.writelines([prefix, result, suffix])
            f.close
    
    def save_row_results(self, model):
        # defined as follows
        # model | stocks taken | stocks lost | damage done | damage recieved | games won | games lost | avg stocks taken per game | avg stocks lost per game | most popular action state | second most popular | third most popular | fourth most popular | fifth most popular

        action_states = []
        for i in range(5):
            if len(self.states) >= (i+1):
                key = max(self.states, key=lambda key: self.states[key])
                action_states.append(key)
                self.states.pop(key)

        ans = model + ' ' + str(self.stocks_taken) + ' ' + str(self.stocks_lost) + ' ' + '{:.2f}'.format(self.damage_done) + ' ' + '{:.2f}'.format(self.damage_recieved) + ' ' + str(self.games_won) + ' ' + str(self.games_lost) + ' ' + '{:.2f}'.format(self.get_average_stocks_taken_game()) + ' ' +  '{:.2f}'.format(self.get_average_stocks_lost_game()) + ' ' + hex(action_states[0]) + ' ' + hex(action_states[1]) + ' ' + hex(action_states[2]) + ' ' + hex(action_states[3]) + ' ' + hex(action_states[4]) + '\n'

        filemode = 'w'
        if os.path.exists('./resultsMatrix.txt'):
            filemode = 'a'
        
        print(filemode)
        with open('resultsMatrix.txt', filemode) as f:
            f.write(ans)
            f.close


    def add_frames(self, frames):
        self.total_frames += frames
        if frames > 1:
            self.skipped_frames += frames - 1

    def add_thinking_time(self, thinking_time):
        self.thinking_time += thinking_time

    def add_metrics(self, state):
        if self.prevState is not None and state.frame >= self.prevState.frame:
            if state.menu == p3.state.Menu.Game and self.prevState.menu == p3.state.Menu.Game:
                self.add_stocks_taken(state)
                self.add_stocks_lost(state)
                self.add_damage_done(state)
                self.add_damage_recieved(state)
                self.track_action_states(state)
            else:
                self.handle_games(state)
        
        self.prevState = copy.deepcopy(state)
    
    def add_stocks_taken(self, state):
        diff = self.prevState.players[0].__dict__['stocks'] - state.players[0].__dict__['stocks']
        if abs(diff) > 1:
            return
        self.stocks_taken += diff
        self.game_stocks_taken += diff

    def add_stocks_lost(self, state):
        diff = self.prevState.players[2].__dict__['stocks'] - state.players[2].__dict__['stocks']
        if abs(diff) > 1:
            return
        self.stocks_lost += diff
        self.game_stocks_lost += diff

    def add_damage_done(self, state):
        diff = state.players[0].__dict__['percent'] - self.prevState.players[0].__dict__['percent']
        if abs(diff) > 80:
            return
        self.damage_done += diff
    
    def add_damage_recieved(self, state):
        diff = state.players[2].__dict__['percent'] - self.prevState.players[2].__dict__['percent']
        if abs(diff) > 80:
            return
        self.damage_recieved += diff

    def handle_games(self, state):
        if self.game_stocks_taken is not 0 or self.game_stocks_lost is not 0:
            if self.game_stocks_taken > self.game_stocks_lost:
                self.games_won += 1
            else:
                self.games_lost += 1
            self.game_stocks_taken = 0
            self.game_stocks_lost = 0

    def get_average_stocks_taken_game(self):
        total_games = self.games_won + self.games_lost
        return self.stocks_taken / total_games if total_games else self.stocks_taken
    
    def get_average_stocks_lost_game(self):
        total_games = self.games_won + self.games_lost
        return self.stocks_lost / total_games if total_games else self.stocks_lost

    def track_action_states(self, state):
        curr_action_state = state.players[2].__dict__['action_state'].value

        if curr_action_state in self.states:
            self.states[curr_action_state] += 1
        else:
            self.states[curr_action_state] = 0
