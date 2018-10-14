package qirkat;

import java.util.ArrayList;

import static qirkat.PieceColor.*;

/** A Player that computes its own moves.
 *  @author Violet Fu
 */
class AI extends Player {

    /** Maximum minimax search depth before going to static evaluation. */
    private static final int MAX_DEPTH = 6;
    /** A position magnitude indicating a win (for white if positive, black
     *  if negative). */
    private static final int WINNING_VALUE = Integer.MAX_VALUE - 1;
    /** A magnitude greater than a normal value. */
    private static final int INFTY = Integer.MAX_VALUE;

    /** A new AI for GAME that will play MYCOLOR. */
    AI(Game game, PieceColor myColor) {
        super(game, myColor);
    }

    @Override
    Move myMove() {
        Main.startTiming();
        Move move = findMove();
        Main.endTiming();
        String msg;
        if (move != null) {
            if (myColor() == WHITE) {
                msg = "White moves ";
            } else {
                msg = "Black moves ";
            }
            msg += move;
            msg += ".";
            System.out.println(msg);
        }
        return move;
    }

    /** Return a move for me from the current position, assuming there
     *  is a move. */
    private Move findMove() {
        Board b = new Board(board());
        if (myColor() == WHITE) {
            findMove(b, MAX_DEPTH, true, 1, -INFTY, INFTY);
        } else {
            findMove(b, MAX_DEPTH, true, -1, -INFTY, INFTY);
        }
        return _lastFoundMove;
    }

    /** The move found by the last call to one of the ...FindMove methods
     *  below. */
    private Move _lastFoundMove;

    /** Find a move from position BOARD and return its value, recording
     *  the move found in _lastFoundMove iff SAVEMOVE. The move
     *  should have maximal value or have value > BETA if SENSE==1,
     *  and minimal value or value < ALPHA if SENSE==-1. Searches up to
     *  DEPTH levels.  Searching at level 0 simply returns a static estimate
     *  of the board value and does not set _lastMoveFound. */
    private int findMove(Board board, int depth, boolean saveMove, int sense,
                         int alpha, int beta) {
        ArrayList<Move> nearby = new ArrayList<>();
        board.getMoves(nearby);
        if (depth == 0) {
            return staticScore(board);
        } else if (nearby.size() > 10 && !nearby.get(0).isJump()) {
            if (saveMove) {
                _lastFoundMove = nearby.get(game().nextRandom(nearby.size()));
            }
            return findMove(board, depth - 1,
                    false, 0 - sense, alpha, beta);
        } else if (nearby.size() == 0) {
            if (depth == MAX_DEPTH) {
                _lastFoundMove = null;
            }
            return (board.whoseMove()
                    == WHITE ? WINNING_VALUE : -WINNING_VALUE);
        } else if (depth == MAX_DEPTH && nearby.size() == 1) {
            _lastFoundMove = nearby.get(0);
            return 0;
        } else if (sense == 1) {
            for (Move move : nearby) {
                board.makeMove(move);
                int result = findMove(board, depth - 1,
                        false, 0 - sense, alpha, beta);
                board.undo();
                if (result >= alpha) {
                    alpha = result;
                    if (saveMove) {
                        _lastFoundMove = move;
                    }
                }
                if (alpha >= beta) {
                    return alpha;
                }
            }
            return alpha;
        } else {
            for (Move move : nearby) {
                board.makeMove(move);
                int result = findMove(board, depth - 1,
                        false, 0 - sense, alpha, beta);
                board.undo();
                if (result <= beta) {
                    beta = result;
                    if (saveMove) {
                        _lastFoundMove = move;
                    }
                }
                if (beta <= alpha) {
                    return beta;
                }
            }
            return beta;
        }
    }


    /** Return a heuristic value for BOARD. */
    private int staticScore(Board board) {
        int result = 0;

        for (int i = 0; i < 5 * 5; i += 1) {
            if (board.get(i) == WHITE) {
                result += 1;
            } else if (board.get(i) == BLACK) {
                result -= 1;
            }
        }
        return result;
    }
}
