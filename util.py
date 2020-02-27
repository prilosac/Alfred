class myDict(dict):
    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

class StateMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        # self.recentlyDied = False
        self.recentlyDied = [False, False, False, False]
    
    # remembers a transition
    def push(self, state):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(state)
        # if len(self.memory) < self.capacity:
        #     self.memory.append(None)
        # self.memory[self.position] = state
        # self.position = (self.position + 1) % self.capacity
    
    def sample(self, batchSize):
        return random.sample(self.memory, batchSize)

    def __len__(self):
        return len(self.memory)
    
    def first(self):
        return self.memory[0]

    def last(self):
        return self.memory[-1]
    
    def died(self, player):
        deaths = [i for i in self.memory if i.players[player].__dict__['action_state'].value <= 0xA]
        if(deaths):
            if not self.recentlyDied[player]:
                self.recentlyDied[player] = True
                return True
        else:
            self.recentlyDied[player] = False

        return False

    def deathCheck(self):
        return self.memory[-2] if len(self) >= 2 else None

def isDying(player):
    # print("dying!")
    print(player.__dict__['action_state'].value, " is")
    if player.__dict__['action_state'].value <= 0xA:
        print("less than")
    else:
        print("larger than")
    print("0xA")
    return player.__dict__['action_state'].value <= 0xA

def chunk(l, n):
  return [l[i:i+n] for i in range(0, len(l), n)]