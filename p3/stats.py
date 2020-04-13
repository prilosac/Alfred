import p3.state
import copy

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

    def __str__(self):
        if not self.total_frames:
            return ''
        frac_thinking = self.thinking_time * 1000 / self.total_frames
        total_games = self.games_won + self.games_lost
        return '\n'.join(
            ['Total Frames: {}'.format(self.total_frames),
             'Skipped: {} | {}%'.format(self.skipped_frames, round(self.skipped_frames/self.total_frames*100, 2)),
             'Average Thinking Time (ms): {:.6f}'.format(frac_thinking), 
             'Stocks Taken: {} | {:.2f}% Dealt'.format(self.stocks_taken, self.damage_done), 'Stocks Lost: {} | {:.2f}% Recieved'.format(self.stocks_lost, self.damage_recieved), 
             'W/L: {}/{} | {:.2f}%'.format(self.games_won, self.games_lost, (self.games_won/total_games if total_games else 0))])

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
            else:
                self.handle_games(state)
        
        self.prevState = copy.deepcopy(state)
    
    def add_stocks_taken(self, state):
        diff = self.prevState.players[0].__dict__['stocks'] - state.players[0].__dict__['stocks']
        self.stocks_taken += diff
        self.game_stocks_taken += diff

    def add_stocks_lost(self, state):
        print(self.prevState.players[2].__dict__['stocks'])
        print(state.players[2].__dict__['stocks'])
        diff = self.prevState.players[2].__dict__['stocks'] - state.players[2].__dict__['stocks']
        self.stocks_lost += diff
        self.game_stocks_lost += diff

    def add_damage_done(self, state):
        print(state.players[0].__dict__['percent'])
        print(self.prevState.players[0].__dict__['percent'])
        self.damage_done += state.players[0].__dict__['percent'] - self.prevState.players[0].__dict__['percent']
    
    def add_damage_recieved(self, state):
        self.damage_recieved += state.players[2].__dict__['percent'] - self.prevState.players[2].__dict__['percent']

    def handle_games(self, state):
        if self.game_stocks_taken is not 0 and self.game_stocks_lost is not 0:
            if self.game_stocks_taken > self.game_stocks_lost:
                self.games_won += 1
            else:
                self.games_lost += 1
            self.game_stocks_taken = 0
            self.game_stocks_lost = 0

    def get_average_stocks_taken_game(self):
        return self.stocks_taken / (self.games_won + self.game_stocks_lost)
    
    def get_average_stocks_taken_game(self):
        return self.stocks_lost / (self.games_won + self.game_stocks_lost)
