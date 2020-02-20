class myDict(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

def isDying(player):
    return player.__dict__['action_state'] <= 0xA