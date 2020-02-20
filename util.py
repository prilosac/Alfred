class myDict(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

def isDying(player):
    return player.__dict__['action_state'].value <= 0xA

def chunk(l, n):
  return [l[i:i+n] for i in range(0, len(l), n)]