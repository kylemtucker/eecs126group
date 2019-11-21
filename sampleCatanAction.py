import numpy as np

def action(self):
    if self.board.settlements == []:
        (x,y) = self.preComp
        self.buy("settlement", x, y)
    elif self.if_can_buy("card"):
        self.buy("card")
    elif self.resources[np.argmax(self.resources)] >= 4:
        rmax, rmin = np.argmax(self.resources), np.argmin(self.resources)
        self.trade(rmax,rmin)
    return

def planBoard(baseBoard):
    x = genRand(0,baseBoard.width+1)
    y = genRand(0,baseBoard.height+1)
    optSettlementLoc = (x,y)
    return optSettlementLoc


def genRand(low,high):
    return np.random.randint(low, high)
