// Copyright (c) 2024 Ole-Christoffer Granmo
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <stdio.h>
#include <stdlib.h>

#ifndef BOARD_DIM
    #define BOARD_DIM 3 // Set to desired board size
#endif

int neighbors[] = {
    -(BOARD_DIM+2) + 1,
    -(BOARD_DIM+2),
    -1,
    1,
    (BOARD_DIM+2),
    (BOARD_DIM+2) - 1
};

struct hex_game {
    int board[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
    int open_positions[BOARD_DIM*BOARD_DIM];
    int number_of_open_positions;
    int moves[BOARD_DIM*BOARD_DIM];
    int connected[(BOARD_DIM+2)*(BOARD_DIM+2)*2];
};

void hg_init(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM+2; ++i) {
        for (int j = 0; j < BOARD_DIM+2; ++j) {
            hg->board[(i*(BOARD_DIM+2) + j)*2] = 0;
            hg->board[(i*(BOARD_DIM+2) + j)*2 + 1] = 0;

            if (i > 0 && i < BOARD_DIM+1 && j > 0 && j < BOARD_DIM+1) {
                hg->open_positions[(i-1)*BOARD_DIM + (j-1)] = i*(BOARD_DIM+2) + j;
            }

            if (i == 0) {
                hg->connected[(i*(BOARD_DIM+2) + j)*2] = 1;
            } else {
                hg->connected[(i*(BOARD_DIM+2) + j)*2] = 0;
            }

            if (j == 0) {
                hg->connected[(i*(BOARD_DIM+2) + j)*2 + 1] = 1;
            } else {
                hg->connected[(i*(BOARD_DIM+2) + j)*2 + 1] = 0;
            }
        }
    }
    hg->number_of_open_positions = BOARD_DIM * BOARD_DIM;
}

int hg_connect(struct hex_game *hg, int player, int position) {
    hg->connected[position*2 + player] = 1;

    if (player == 0 && position / (BOARD_DIM+2) == BOARD_DIM) {
        return 1;
    }
    if (player == 1 && position % (BOARD_DIM+2) == BOARD_DIM) {
        return 1;
    }

    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (hg->board[neighbor*2 + player] && !hg->connected[neighbor*2 + player]) {
            if (hg_connect(hg, player, neighbor)) {
                return 1;
            }
        }
    }
    return 0;
}

int hg_winner(struct hex_game *hg, int player, int position) {
    for (int i = 0; i < 6; ++i) {
        int neighbor = position + neighbors[i];
        if (hg->connected[neighbor*2 + player]) {
            return hg_connect(hg, player, position);
        }
    }
    return 0;
}

int hg_place_piece_randomly(struct hex_game *hg, int player) {
    int random_empty_position_index = rand() % hg->number_of_open_positions;
    int empty_position = hg->open_positions[random_empty_position_index];

    if (player == 0) {
        hg->board[empty_position * 2] = 1;      // Player 1
    } else {
        hg->board[empty_position * 2 + 1] = 2;  // Player 2
    }

    hg->moves[BOARD_DIM*BOARD_DIM - hg->number_of_open_positions] = empty_position;
    hg->open_positions[random_empty_position_index] = hg->open_positions[hg->number_of_open_positions - 1];
    hg->number_of_open_positions--;

    return empty_position;
}

int hg_full_board(struct hex_game *hg) {
    return (hg->number_of_open_positions == 0);
}

void hg_print_board(struct hex_game *hg) {
    for (int i = 0; i < BOARD_DIM; ++i) {
        for (int j = 0; j < i; j++) {
            printf(" ");
        }

        for (int j = 0; j < BOARD_DIM; ++j) {
            int pos = ((i+1)*(BOARD_DIM+2) + j + 1)*2;
            if (hg->board[pos] == 1) {
                printf(" X");
            } else if (hg->board[pos + 1] == 2) {
                printf(" O");
            } else {
                printf(" .");
            }
        }
        printf("\n");
    }
}

void write_csv_header(FILE *file) {
    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            fprintf(file, "%d_%d,", i, j); // Use coordinates for column names
        }
    }
    fprintf(file, "Winner\n");
}

void save_game_data(struct hex_game *hg, int winner, FILE *file) {
    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            int pos = (i * (BOARD_DIM + 2) + j) * 2;
            if (hg->board[pos] == 1) {
                fprintf(file, "X,");
            } else if (hg->board[pos + 1] == 2) {
                fprintf(file, "O,");
            } else {
                fprintf(file, ".,");
            }
        }
    }
    fprintf(file, "%d\n", (winner == 1) ? 1 : 0);
}

void save_partial_game_data(struct hex_game *hg, int winner, FILE *file, int moves_to_remove) {
    struct hex_game temp = *hg;  // Copy current game state

    // Remove last `moves_to_remove` moves
    for (int i = 0; i < moves_to_remove && temp.number_of_open_positions < BOARD_DIM * BOARD_DIM; ++i) {
        int last_move_index = BOARD_DIM * BOARD_DIM - temp.number_of_open_positions - 1;
        int last_move = temp.moves[last_move_index];

        // Determine which player's move is being removed
        if (temp.board[last_move * 2] == 1) {
            temp.board[last_move * 2] = 0;          // Remove X
        } else if (temp.board[last_move * 2 + 1] == 2) {
            temp.board[last_move * 2 + 1] = 0;      // Remove O
        }

        temp.number_of_open_positions++;
        temp.open_positions[temp.number_of_open_positions - 1] = last_move;
    }

    // Save resulting board state
    for (int i = 1; i <= BOARD_DIM; ++i) {
        for (int j = 1; j <= BOARD_DIM; ++j) {
            int pos = (i * (BOARD_DIM + 2) + j) * 2;
            if (temp.board[pos] == 1) {
                fprintf(file, "X,");
            } else if (temp.board[pos + 1] == 2) {
                fprintf(file, "O,");
            } else {
                fprintf(file, ".,");
            }
        }
    }
    fprintf(file, "%d\n", (winner == 1) ? 1 : 0);
}

int main() {
    struct hex_game hg;

    FILE *file_complete = fopen("hex_game_data_complete.csv", "w");
    FILE *file_2_moves_before = fopen("hex_game_data_2_moves_before.csv", "w");
    FILE *file_5_moves_before = fopen("hex_game_data_5_moves_before.csv", "w");

    write_csv_header(file_complete);
    write_csv_header(file_2_moves_before);
    write_csv_header(file_5_moves_before);

    // For different board sizes, vary the threshold of empty cells
    #define EMPTY_CELL_THRESHOLD ((int)(BOARD_DIM * BOARD_DIM * 0.2))

    int winner = -1;
    int total_games = 10000000; 
    int target_valid_games = 2000;
    int valid_games = 0;
    int skipped_games = 0;
    int x_wins = 0;
    int y_wins = 0;
    int total_moves = 0;

    for (int game = 0; game < total_games; ++game) {
        hg_init(&hg);

        int player = 0;
        int moves_played = 0;

        while (!hg_full_board(&hg)) {
            int position = hg_place_piece_randomly(&hg, player);
            moves_played++;
            if (hg_winner(&hg, player, position)) {
                winner = (player == 0) ? 1 : 2;
                break;
            }
            player = 1 - player;
        }

        if (hg.number_of_open_positions < EMPTY_CELL_THRESHOLD) {
            // Skip if fewer empty positions than threshold
            skipped_games++;
            continue;
        }

        // Balance data to avoid bias
        if (winner == 1 && (double)x_wins / (x_wins + y_wins + 1) > 0.55) {
            skipped_games++;
            continue;
        }

        if (winner == 2 && (double)y_wins / (x_wins + y_wins + 1) > 0.55) {
            skipped_games++;
            continue;
        }

        if (winner == 1) {
            x_wins++;
        } else if (winner == 2) {
            y_wins++;
        }

        save_game_data(&hg, winner, file_complete);
        save_partial_game_data(&hg, winner, file_2_moves_before, 2);
        save_partial_game_data(&hg, winner, file_5_moves_before, 5);

        total_moves += moves_played;
        valid_games++;

        printf("\nValid Game %d (Player %d wins):\n", valid_games, winner);
        hg_print_board(&hg);

        if (valid_games >= target_valid_games) {
            break;
        }
    }

    fclose(file_complete);
    fclose(file_2_moves_before);
    fclose(file_5_moves_before);

    printf("\nTotal valid games: %d\n", valid_games);
    printf("Skipped games: %d\n", skipped_games);
    printf("Player X (1) wins: %d\n", x_wins);
    printf("Player Y (2) wins: %d\n", y_wins);
    printf("Average moves per valid game: %.2f\n", (double)total_moves / valid_games);

    return 0;
}
