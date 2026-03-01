# import torch for tensors
import torch

# board object
class Board:
    # create board
    def __init__(self):
        # use gpu for faster operations
        self.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

        # stores bit boards for each player
        self.player1_bits = torch.zeros((6, 7), dtype=torch.float32, device=self.device)
        self.player2_bits = torch.zeros((6, 7), dtype=torch.float32, device=self.device)

        # tracks current player's turn (1 or 2)
        self.current_player = 1


    # check horizontal 4 in a row
    def check_hor(self, player, x, y):
        bits = self.player1_bits if player == 1 else self.player2_bits

        start = max(y - 3, 0)
        end = min(y + 4, 7)

        row_slice = bits[x, start:end]

        for i in range(row_slice.size(0) - 3):
            if row_slice[i:i+4].all():
                return True

        return False


    # check vertical 4 in a row
    def check_vert(self, player, x, y):
        bits = self.player1_bits if player == 1 else self.player2_bits

        start = max(x - 3, 0)
        end = min(x + 4, 6)

        col_slice = bits[start:end, y]

        for i in range(col_slice.size(0) - 3):
            if col_slice[i:i+4].all():
                return True

        return False


    # check main diagonal (\) 4 in a row
    def check_diag1(self, player, x, y):
        bits = self.player1_bits if player == 1 else self.player2_bits

        i, j = x, y
        while i > 0 and j > 0:
            i -= 1
            j -= 1

        diag_slice = []
        while i < 6 and j < 7:
            diag_slice.append(bits[i, j])
            i += 1
            j += 1

        diag_slice = torch.stack(diag_slice)

        for k in range(diag_slice.size(0) - 3):
            if diag_slice[k:k+4].all():
                return True

        return False


    # check other diagonal (/) 4 in a row
    def check_diag2(self, player, x, y):
        bits = self.player1_bits if player == 1 else self.player2_bits

        i, j = x, y
        while i < 5 and j > 0:
            i += 1
            j -= 1

        diag_slice = []
        while i >= 0 and j < 7:
            diag_slice.append(bits[i, j])
            i -= 1
            j += 1

        diag_slice = torch.stack(diag_slice)

        for k in range(diag_slice.size(0) - 3):
            if diag_slice[k:k+4].all():
                return True

        return False


    # full check for any winning 4 in a row
    def full_check(self, player, x, y):
        return (self.check_hor(player, x, y) or
                self.check_vert(player, x, y) or
                self.check_diag1(player, x, y) or
                self.check_diag2(player, x, y))


    def contained_cells(self):
        return (self.player1_bits + self.player2_bits) != 0


    # checks if the board is full
    def is_full(self):
        return self.contained_cells().all()


    # drops a coin in the given column
    def make_move(self, player, col):
        taken = self.contained_cells()[:, col]

        if taken.all():
            return False

        for i in range(6):
            if not taken[i]:
                if player == 1:
                    self.player1_bits[i, col] = 1
                else:
                    self.player2_bits[i, col] = 1

                win = self.full_check(player, i, col)
                self.current_player = 3 - player
                return win

        return False