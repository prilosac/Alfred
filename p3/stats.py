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

    def __str__(self):
        if not self.total_frames:
            return ''
        frac_thinking = self.thinking_time * 1000 / self.total_frames
        return '\n'.join(
            ['Total Frames: {}'.format(self.total_frames),
             'Skipped: {} | {}%'.format(self.skipped_frames, round(self.skipped_frames/self.total_frames*100, 2)),
             'Average Thinking Time (ms): {:.6f}'.format(frac_thinking)])

    def add_frames(self, frames):
        self.total_frames += frames
        if frames > 1:
            self.skipped_frames += frames - 1

    def add_thinking_time(self, thinking_time):
        self.thinking_time += thinking_time

    def add_metrics(self, state):
        if self.prevState is not None:
            self.add_stocks_taken(state)
            self.add_stocks_lost(state)
            self.add_damage_done(state)
            self.add_damage_recieved(state)
        
        self.prevState = state
    
    def add_stocks_taken(self, state):
        self.stocks_taken += self.prevState.players[0].__dict__['stocks'] - state.players[0].__dict__['stocks']

    def add_stocks_lost(self, state):
        self.stocks_lost += self.prevState.players[3].__dict__['stocks'] - state.players[3].__dict__['stocks']

    def add_damage_done(self, state):
        self.damage_done += state.players[0].__dict__['percent'] - self.prevState.players[0].__dict__['percent']
    
    def add_damage_recieved(self, state):
        self.damage_recieved += state.players[3].__dict__['percent'] - self.prevState.players[3].__dict__['percent']